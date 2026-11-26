"""Speak streamable-HTTP to a remote MCP server, over httpx.

This is the m10e transport: one URL, ``POST`` per JSON-RPC message, and the
framing that turns a response body — a JSON document or an SSE stream — into the
frames :class:`~bevel.mcp.jsonrpc.JsonRpcChannel` pairs. It satisfies
:class:`~bevel.mcp.jsonrpc.DuplexStream` and nothing wider, exactly as
:mod:`~bevel.mcp.stdio` does, and the pairing layer is shared with it unchanged:
this module imports the channel, not a variant of it.

**The specification, and the assumptions this module makes where it is quiet.**
Everything below is what the 2025-06-18 "Transports" revision requires; the
bullets marked *assumption* are places the document leaves a choice and bevel
makes it explicitly rather than by accident. Each is pinned by a test in
``tests/unit/test_mcp_http.py`` against ``tests/fixtures/mcp_server.py --http``,
which enforces them by refusing non-compliant requests rather than by echoing
headers back for a test to eyeball — a client that sends ``Accept`` only when it
remembers satisfies an echo and fails the refusal.

* One ``POST`` per JSON-RPC message, ``Content-Type: application/json``.
* ``Accept`` lists **both** ``application/json`` and ``text/event-stream`` on
  every request, because the server picks either shape for any given reply and
  the client cannot know in advance which.
* A notification or a response is answered ``202 Accepted`` with no body. A
  request is answered ``200`` with either a JSON body or an SSE stream.
* ``Mcp-Session-Id`` is captured from the ``initialize`` response and replayed
  verbatim on every later request. ``initialize`` itself carries none.
* ``MCP-Protocol-Version`` is sent on every request once the version is
  negotiated, and omitted before it — the server cannot have agreed to a version
  it has not yet answered with.
* ``400`` means a required header or session id is missing or wrong; ``404``
  means the server dropped the session. A dropped session is fatal, never
  retried: re-sending a ``tools/call`` would run the tool twice.
* ``DELETE`` ends the session, sent once by :meth:`HttpTransport.close`.
* *Assumption:* resumability (``Last-Event-ID`` and event ids) is not
  implemented. It exists to survive a dropped stream mid-response; bevel's
  requests are short and a dropped stream fails the call, which is the same
  outcome the spec's own guidance reaches. Replaying a ``tools/call`` from a
  buffer would be worse than failing it.
* *Assumption:* the server-initiated ``GET`` stream is not opened. bevel has no
  consumer for server-initiated notifications today, and an idle ``GET`` held
  open for the life of the session is a connection with no purpose.
* *Assumption:* one ``POST`` is one request. bevel never sends a batch, so a
  response body carrying several JSON objects is only ever produced by an SSE
  stream of notifications plus the reply.

**Why the POST happens in :meth:`write_frame` and not in a background task.**
stdio needs a reader task because a pipe fills up whether or not anyone is
reading it, and a server blocked in ``write`` deadlocks a client blocked in
``read``. HTTP has no such buffer: the server's reply is the body of the request
we just sent, and the socket is drained by httpx as it arrives. So a request's
POST runs to completion inside ``write_frame``, its frames are pushed onto the
queue, and :meth:`read_frame` carves them off it. There is no pump to start, no
buffer to keep draining, and no task to leak on teardown.

**What this module is not.** It speaks ``initialize`` and ``tools/list`` on the
wire and stops there, exactly as stdio does. It does not know what a ``ToolSpec``
is, does not namespace remote tool names, and does not turn content blocks into a
``ToolResult`` — that is :mod:`~bevel.mcp.bridge`. It also does not retry: a
transport that silently re-sent a request would be re-running a tool the model
already asked for.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

import httpx

from bevel.mcp.jsonrpc import JsonRpcChannel, RemoteError, StreamClosedError

# The taxonomy and the version list live in stdio, which shipped first. They are
# imported rather than duplicated because "the MCP transport failed" must be one
# base class for the TUI to guard a session with one ``except``, and a second
# SUPPORTED_PROTOCOL_VERSIONS is a second list to forget to update.
from bevel.mcp.stdio import (
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_HANDSHAKE_TIMEOUT,
    SUPPORTED_PROTOCOL_VERSIONS,
    McpHandshakeError,
    McpTimeoutError,
    McpTransportError,
)

#: How long to wait for the TCP connection and TLS handshake before giving up.
#: Separate from the read budget because "nothing is listening" is answered in
#: milliseconds by any honest stack, and waiting a minute for it is a bug report
#: that says "bevel hangs" instead of "the URL is wrong".
DEFAULT_CONNECT_TIMEOUT: Final[float] = 15.0

#: What the spec requires the client to offer, in both directions.
ACCEPT: Final[str] = "application/json, text/event-stream"
CONTENT_TYPE: Final[str] = "application/json"

#: Bytes of server response text kept for error messages, for the same reason
#: stdio bounds its stderr: the server decides how much it writes.
_BODY_CAP: Final[int] = 4096
_DATA_PREFIX: Final[str] = "data:"


class McpConnectError(McpTransportError):
    """The URL is there in the config and the server is not there at the other end.

    Distinct from a timeout because the remedies differ: this one is a wrong URL,
    a server that is not running, or a firewall, and no increase to a timeout
    fixes any of them.
    """


class McpSessionExpiredError(McpTransportError):
    """The server answered ``404``: the session id we hold is no longer one it knows.

    Named for what it means rather than for the status code, because the action
    it implies is "start a new session", not "send the request again". Retrying a
    ``tools/call`` against a server that has forgotten it would run the tool a
    second time.
    """


class HttpTransport:
    """A remote MCP server spoken to over streamable-HTTP.

    Lifecycle: :meth:`start`, then :meth:`call` and :meth:`notify`, then
    :meth:`close`. There is no process to kill, so teardown is one ``DELETE``
    and closing the connection pool — and, as in stdio, :meth:`close` is safe to
    call twice because :meth:`start` calls it on every failure path.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> None:
        # Checked here, not on first use, so a config typo is reported when the
        # server is named rather than after the TUI has drawn itself around it.
        if not url.startswith(("http://", "https://")):
            raise McpTransportError(
                f"the MCP server URL {url!r} is not usable: it must start with http:// or "
                f"https:// so the transport knows which protocol to speak."
            )
        self._url = url
        self._extra_headers = dict(headers) if headers is not None else {}
        self._connect_timeout = connect_timeout
        self._handshake_timeout = handshake_timeout
        self._call_timeout = call_timeout

        # No read timeout: every wait is bounded by asyncio.wait_for at the call
        # site, which is what lets the error name the stage and the method. An
        # httpx read timeout would surface as an anonymous ReadTimeout instead.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=connect_timeout),
            follow_redirects=False,
        )
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._wakeup = asyncio.Event()
        # Two concurrent call()s mean two concurrent read_frame()s, because
        # JsonRpcChannel.call pumps receive() itself. The lock makes "is there a
        # frame? is the stream dead? then wait" one decision, so a frame pushed
        # between the check and the wait cannot be missed.
        self._read_lock = asyncio.Lock()
        self._failure: BaseException | None = None
        self._closed = False
        self._delete_sent = False
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._server_info: dict[str, Any] = {}
        self._server_text = ""
        self._channel = JsonRpcChannel(self)

    # --- the two methods DuplexStream asks for ------------------------------

    async def read_frame(self) -> bytes | None:
        """Return the next message the server sent, or ``None`` once it is gone.

        A frame is one JSON-RPC message: a whole JSON body, or one ``data:``
        value from an SSE stream. Either way it arrives here as the ``bytes`` the
        channel parses, which is what makes the channel shared with stdio rather
        than forked — stdio's frame is a line off a pipe and this one is an event
        off a response, and neither the channel nor the parser needs to know.
        """
        async with self._read_lock:
            while True:
                if not self._queue.empty():
                    return self._queue.get_nowait()
                if self._failure is not None:
                    raise self._failure
                if self._closed:
                    return None
                self._wakeup.clear()
                await self._wakeup.wait()

    async def write_frame(self, payload: bytes) -> None:
        """Send one message as one ``POST``, and queue whatever came back.

        A request waits for its response here; a notification is content with the
        ``202`` the spec gives it. The distinction is read out of the payload
        rather than passed in, because :class:`~bevel.mcp.jsonrpc.JsonRpcChannel`
        hands over bare bytes and the delimiter — here, the presence of an ``id``
        — is the transport's business.
        """
        if self._closed:
            raise StreamClosedError("the transport is closed; nothing can be written")
        is_request = _is_request(payload)
        response = await self._post(payload, expect_body=is_request)

        if response.status_code == 202:
            # A notification, accepted, and answered with nothing. Queuing a
            # frame for it would hand the channel a phantom.
            return
        captured = response.headers.get("mcp-session-id")
        if captured and self._session_id is None:
            self._session_id = captured
        self._queue_body(response)

    # --- lifecycle ----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        """Always ``None``: a remote server is not our child.

        Kept as a property rather than omitted so ``ServerReport`` can ask either
        transport the same question and print "no process to leak" honestly for
        one and a real pid for the other.
        """
        return None

    @property
    def url(self) -> str:
        """The endpoint this transport speaks to."""
        return self._url

    @property
    def channel(self) -> JsonRpcChannel:
        """The pairing layer, exposed so a caller can see undelivered frames."""
        return self._channel

    @property
    def server_info(self) -> Mapping[str, Any]:
        """The ``serverInfo`` the server named itself with in ``initialize``."""
        return dict(self._server_info)

    @property
    def protocol_version(self) -> str | None:
        """The version the two ends agreed on, or ``None`` before a handshake."""
        return self._protocol_version

    @property
    def session_id(self) -> str | None:
        """The ``Mcp-Session-Id`` the server issued, or ``None`` if it issued none."""
        return self._session_id

    @property
    def stderr_tail(self) -> str:
        """The tail of the last error body the server returned.

        stdio has a stderr pipe; an HTTP server has a response body, and a
        ``400`` from a server that says *why* is the same kind of evidence. It is
        what turns "the call failed" into something a user can act on, so
        :mod:`~bevel.mcp.bridge` quotes it the same way for both transports.
        """
        return self._server_text

    async def start(self) -> list[dict[str, Any]]:
        """Negotiate the protocol version and return the advertised tools.

        ``initialize`` → ``notifications/initialized`` → ``tools/list``, the same
        three messages stdio sends, in the same order, for the same reason: a
        strict server refuses any request that arrives before it has been told the
        client is ready.
        """
        try:
            result = await self._handshake(
                "initialize",
                {
                    "protocolVersion": SUPPORTED_PROTOCOL_VERSIONS[0],
                    "capabilities": {},
                    "clientInfo": {"name": "bevel", "version": _client_version()},
                },
                "the server's initialize result",
            )
            self._agree_on_version(result)
            await self._channel.notify("notifications/initialized")
            listed = await self._handshake("tools/list", {}, "the server's tools/list result")
            return _tool_list(listed, self._describe())
        except BaseException:
            await self.close()
            raise
        # The caller owns the transport from here; close() is its to call.

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | Sequence[Any] | None = None,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - a per-call override of the configured budget, not a local deadline
    ) -> Any:
        """Send one request and return its result.

        Raises :class:`~bevel.mcp.jsonrpc.RemoteError` when the server answers
        with an error object — an ``isError`` tool failure is a *result*, not a
        broken connection — and :class:`McpTimeoutError` naming the call stage
        when the reply does not come.
        """
        if self._closed:
            raise StreamClosedError(f"the transport is closed; {method!r} was not sent")
        budget = self._call_timeout if timeout is None else timeout
        try:
            return await asyncio.wait_for(self._channel.call(method, params), budget)
        except TimeoutError as exc:
            raise McpTimeoutError(
                f"call timed out after {budget:.1f}s: the MCP server at {self._describe()} "
                f"accepted the connection but never answered {method!r}. {self._server_clause()}"
            ) from exc

    async def notify(
        self,
        method: str,
        params: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> None:
        """Send a notification. Returns once written; waits for no reply."""
        if self._closed:
            raise StreamClosedError(f"the transport is closed; {method!r} was not sent")
        await self._channel.notify(method, params)

    async def close(self) -> None:
        """End the session and release the connection pool. Safe twice.

        The ``DELETE`` is sent only when a session id was actually issued — a
        server that never gave us one has nothing to terminate — and only once,
        because :meth:`start` calls this on every failure path and a caller that
        also closes in a ``finally`` must not leave two requests behind. Errors
        from the ``DELETE`` are swallowed: a server already gone is the outcome
        this method was asked to produce.
        """
        if self._closed:
            return
        self._closed = True
        if self._session_id is not None and not self._delete_sent:
            self._delete_sent = True
            with contextlib.suppress(httpx.HTTPError, McpTransportError, RuntimeError):
                await self._client.delete(self._url, headers=self._headers())
        with contextlib.suppress(Exception):
            await self._client.aclose()
        # Anyone parked in read_frame must see the end rather than wait for a
        # response that will now never be pushed.
        self._wakeup.set()

    # --- internals ----------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Every mandatory header, plus whatever the config added.

        The configured values are applied last so a server that needs a
        non-standard ``Accept`` can still be reached, but nothing else is
        negotiable: dropping ``Accept`` or the session id is a spec violation the
        server answers with a ``400`` we would then have to diagnose remotely.
        """
        headers = {"Accept": ACCEPT, "Content-Type": CONTENT_TYPE, **self._extra_headers}
        if self._protocol_version is not None:
            headers["MCP-Protocol-Version"] = self._protocol_version
        if self._session_id is not None:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, payload: bytes, *, expect_body: bool) -> httpx.Response:
        """One POST, with the status codes turned into typed errors.

        Nothing retries. A ``500`` from a server that may have already run the
        tool is exactly the case a transport must not paper over.
        """
        try:
            response = await self._client.post(self._url, content=payload, headers=self._headers())
        except httpx.HTTPError as exc:
            self._failure = McpConnectError(
                f"connection failed: the MCP server at {self._describe()} could not be reached "
                f"({type(exc).__name__}: {exc}). Check the url in its [mcp.servers] entry and "
                f"that the server is running."
            )
            self._wakeup.set()
            raise self._failure from exc

        if response.status_code == 202:
            return response
        if response.status_code == 404:
            raise self._refusal("session expired", response)
        if response.status_code in (400, 401, 403, 405, 406):
            raise self._refusal("refused the request", response)
        if response.status_code >= 400:
            raise self._refusal(f"answered HTTP {response.status_code}", response)
        if expect_body and not response.content:
            raise McpTransportError(
                f"the MCP server at {self._describe()} answered {response.status_code} with an "
                f"empty body, so there is no reply to read."
            )
        return response

    def _refusal(self, what: str, response: httpx.Response) -> McpTransportError:
        """A typed error quoting what the server said, not just the status it used.

        A ``400`` from a spec-compliant server names the header or the session id
        that was wrong. Dropping that body is what makes an MCP client debugging
        session take an afternoon.
        """
        body = response.text[:_BODY_CAP]
        self._server_text = body
        if response.status_code == 404:
            return McpSessionExpiredError(
                f"the MCP server at {self._describe()} answered HTTP 404: the session "
                f"{self._session_id!r} it issued is no longer one it recognises. bevel does not "
                f"replay a request against a server that has forgotten it, because a "
                f"'tools/call' would run twice. Start a new session. Server said: {body!r}"
            )
        return McpTransportError(
            f"the MCP server at {self._describe()} {what}: HTTP {response.status_code}. "
            f"Server said: {body!r}"
        )

    def _queue_body(self, response: httpx.Response) -> None:
        """Push each JSON-RPC message in the response onto the read queue.

        Two shapes, chosen by the server per response. A JSON body is one message
        (or, for a server that answers a request with notifications in front of
        the reply, several newline-separated ones — split on newlines so each is
        still handed over as its own frame). An SSE body is one message per
        ``data:`` value, with the ``event:``/``id:`` fields and ``:`` comments
        discarded, which is what the SSE framing rules say a reader does and what
        makes the frame here byte-for-byte the frame stdio's reader produces.
        """
        content_type = response.headers.get("content-type", "")
        text = response.text
        if "text/event-stream" in content_type:
            frames = _sse_events(text)
        else:
            frames = [line for line in text.splitlines() if line.strip()]
        for frame in frames:
            self._queue.put_nowait(frame.encode("utf-8"))
        self._wakeup.set()

    async def _handshake(self, method: str, params: Mapping[str, Any], waiting_for: str) -> Any:
        """One request during :meth:`start`, under the handshake budget."""
        try:
            return await asyncio.wait_for(
                self._channel.call(method, params), self._handshake_timeout
            )
        except TimeoutError as exc:
            raise McpTimeoutError(
                f"handshake timed out after {self._handshake_timeout:.1f}s: the MCP server at "
                f"{self._describe()} accepted the connection but never answered {method!r}, "
                f"while waiting for {waiting_for}. {self._server_clause()}"
            ) from exc
        except RemoteError as exc:
            raise McpHandshakeError(
                f"handshake failed: the MCP server at {self._describe()} rejected {method!r} "
                f"with code {exc.code}: {exc.message}. {self._server_clause()}"
            ) from exc

    def _agree_on_version(self, result: Any) -> None:
        """Accept a version we support; refuse a peer that will not negotiate.

        The rules are stdio's, verbatim, and deliberately so: a server must not
        get to negotiate differently depending on how it is reached.
        """
        supported = ", ".join(SUPPORTED_PROTOCOL_VERSIONS)
        if not isinstance(result, dict):
            raise McpHandshakeError(
                f"handshake failed: initialize returned {type(result).__name__} from "
                f"{self._describe()}, expected an object with protocolVersion and capabilities."
            )
        version = result.get("protocolVersion")
        if not isinstance(version, str):
            raise McpHandshakeError(
                f"handshake failed: the server at {self._describe()} omitted protocolVersion "
                f"from its initialize result, so there is no protocol to speak. bevel "
                f"implements: {supported}."
            )
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpHandshakeError(
                f"handshake failed: the server at {self._describe()} answered with "
                f"protocolVersion {version!r}, which bevel does not implement. bevel "
                f"implements: {supported}."
            )
        self._protocol_version = version
        info = result.get("serverInfo")
        self._server_info = dict(info) if isinstance(info, dict) else {}

    def _describe(self) -> str:
        # The URL, unredacted. Credentials in a URL are the operator's choice and
        # hiding them here would make the failure unreproducible; the config layer
        # is where a secret belongs, in ``env`` or a header, not in the path.
        return self._url

    def _server_clause(self) -> str:
        tail = self._server_text.strip()
        return (
            f"Last response body from the server: {tail!r}"
            if tail
            else "The server returned no response body."
        )


def _is_request(payload: bytes) -> bool:
    """Whether these bytes are a request (await a reply) or a notification (do not).

    Read from the payload rather than passed in because ``DuplexStream`` is two
    methods and no more, and widening it to carry the distinction would be the
    pairing layer learning about transports — the exact coupling the protocol
    exists to prevent.
    """
    try:
        frame = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True
    return isinstance(frame, dict) and "id" in frame


def _sse_events(text: str) -> list[str]:
    """Each SSE event's data, as one string per event.

    An event is terminated by a blank line; its ``data:`` lines are joined with
    ``\\n``. ``event:``, ``id:`` and ``retry:`` fields and ``:`` comments are
    dropped, which is the framing rule rather than a shortcut: a client that
    treated every line as data would hand the channel ``event: message`` as a
    JSON-RPC message, and the fixture sends that field precisely to catch it.
    """
    events: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            if current:
                events.append("\n".join(current))
                current = []
            continue
        if raw.startswith(":"):
            continue
        field, _, value = raw.partition(":")
        if field.strip() == "data":
            current.append(value[1:] if value.startswith(" ") else value)
    if current:
        events.append("\n".join(current))
    return events


def _tool_list(result: Any, described_url: str) -> list[dict[str, Any]]:
    """Pull ``tools`` out of a ``tools/list`` result, or say why there is none."""
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        raise McpHandshakeError(
            f"handshake failed: tools/list returned {result!r:.200} from {described_url}, "
            f"which has no 'tools' array."
        )
    return [tool for tool in result["tools"] if isinstance(tool, dict)]


def _client_version() -> str:
    from bevel import __version__  # noqa: PLC0415 - a module-scope import would cycle bevel -> mcp

    return __version__


__all__ = [
    "HttpTransport",
    "McpConnectError",
    "McpSessionExpiredError",
]

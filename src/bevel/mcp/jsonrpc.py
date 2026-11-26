"""JSON-RPC 2.0 envelopes, ids, error codes and request/response pairing.

This is the part of the MCP client that can be proven correct without a server.
It owns no file descriptor, no socket, no subprocess and no asyncio transport:
it turns bytes into typed envelopes and routes replies back to the coroutine
that asked. m10b hands it a pipe; everything here is testable by handing it a
queue of ``bytes``.

Two rules shape the whole module.

**Nothing is ever silently dropped.** A frame we cannot understand is still a
frame that arrived, and the caller waiting on the other side of it does not
know the difference between "the peer is thinking" and "the reply came back
broken three seconds ago". So a parse failure becomes an :class:`ErrorObject`
carrying the spec's own code, an unmatched id becomes an
:class:`UnmatchedResponse` in :attr:`JsonRpcChannel.reports`, and a closed
stream fails *every* outstanding request rather than only the one the reader
happened to be awaiting.

**The spec's codes are the contract, not our convenience.** ``-32700`` means
the bytes were not JSON; ``-32600`` means they were JSON but not a legal
object. Collapsing those into one "bad response" would throw away the
distinction that tells you whether to blame your framing or your peer.

No MCP vocabulary lives here on purpose. There is no ``initialize``, no
``tools/list``, no notion of a tool or a server. Those are m10b (transport and
handshake) and m10c (the bridge into bevel's tool seam); this module would
compile unchanged against any JSON-RPC peer, which is what keeps it testable.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, cast, runtime_checkable

__all__ = [
    "INTERNAL_ERROR",
    "INVALID_PARAMS",
    "INVALID_REQUEST",
    "JSONRPC_VERSION",
    "MAX_SAFE_INTEGER",
    "METHOD_NOT_FOUND",
    "PARSE_ERROR",
    "DuplexStream",
    "ErrorObject",
    "IdAllocator",
    "IdExhaustedError",
    "InboundNotification",
    "InboundRequest",
    "Incoming",
    "JsonRpcChannel",
    "JsonRpcError",
    "MalformedFrame",
    "Notification",
    "PendingRequest",
    "ProtocolViolationError",
    "RemoteError",
    "Request",
    "Response",
    "StreamClosedError",
    "UnmatchedResponse",
    "parse_frame",
]

JSONRPC_VERSION: Final = "2.0"

PARSE_ERROR: Final = -32700
"""The bytes were not JSON at all."""

INVALID_REQUEST: Final = -32600
"""Valid JSON, but not a legal request/response object."""

METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603

#: Largest integer every JSON parser round-trips exactly. JSON numbers are
#: IEEE-754 doubles, so an id above this can arrive back as a *different*
#: number and be delivered to the wrong caller. The allocator refuses to go
#: there rather than discovering it in production.
MAX_SAFE_INTEGER: Final = 2**53 - 1

_RAW_CLIP: Final = 300
"""How much of a broken frame to keep in an error message.

Long enough to diagnose a truncated or mis-encoded frame, short enough that a
peer dumping a megabyte onto the wire cannot turn an exception message into a
megabyte of log.
"""


class JsonRpcError(Exception):
    """Base for everything this module raises.

    A single base so a caller can guard the whole seam with one ``except`` and
    still branch on the specific failure when it needs to.
    """


class ProtocolViolationError(JsonRpcError):
    """Something we were asked to send is not sendable, or a peer sent bytes
    we cannot act on.

    Carries the JSON-RPC code it maps onto rather than being a bare
    ``ValueError``: the caller has to decide whether *it* misbuilt the request
    (a bug to fix) or the peer misbehaved (a thing to report), and the code is
    what makes that decision.
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.data = data


class RemoteError(JsonRpcError):
    """The peer answered with an ``error`` object.

    Not an exception the caller should treat as "the transport broke" -- the
    transport worked, and the peer said no. m10c turns this into a failed
    :class:`~bevel.core.tools.base.ToolResult`, never a crash in the loop.
    """

    def __init__(
        self,
        code: int,
        message: str,
        *,
        data: Any = None,
        request_id: str | int | None = None,
        method: str = "",
    ) -> None:
        super().__init__(
            f"JSON-RPC call {method!r} failed with code {code}: {message}"
            + (f" (data: {data!r})" if data is not None else "")
        )
        self.code = code
        self.message = message
        self.data = data
        self.request_id = request_id
        self.method = method


class IdExhaustedError(JsonRpcError):
    """The allocator ran out of ids.

    Raised rather than wrapping around: reusing an id that is still in flight
    delivers one peer's answer to a different caller's conversation, which is
    a far worse bug than a client that refuses to send another request.
    """


class StreamClosedError(JsonRpcError):
    """The peer half of the stream ended while requests were outstanding."""


@dataclass(frozen=True)
class ErrorObject:
    """The ``error`` member of a JSON-RPC response.

    Frozen because it is a value read off the wire and passed around; the
    caller that receives it must not be able to rewrite what the peer said.
    """

    code: int
    message: str
    data: Any = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            wire["data"] = self.data
        return wire

    @classmethod
    def parse_error(cls, message: str, data: Any = None) -> ErrorObject:
        return cls(PARSE_ERROR, message, data)

    @classmethod
    def invalid_request(cls, message: str, data: Any = None) -> ErrorObject:
        return cls(INVALID_REQUEST, message, data)

    @classmethod
    def method_not_found(cls, message: str, data: Any = None) -> ErrorObject:
        return cls(METHOD_NOT_FOUND, message, data)

    @classmethod
    def invalid_params(cls, message: str, data: Any = None) -> ErrorObject:
        return cls(INVALID_PARAMS, message, data)

    @classmethod
    def internal_error(cls, message: str, data: Any = None) -> ErrorObject:
        return cls(INTERNAL_ERROR, message, data)


def _check_method(method: Any) -> None:
    if not isinstance(method, str) or not method:
        raise ProtocolViolationError(
            INVALID_PARAMS,
            f"method must be a non-empty string, got {_json_type_name(method)}",
        )


def _check_params(params: Any) -> None:
    # ``params`` is model-authored by the time m10c gets here, and Python's
    # ``str`` satisfies ``Sequence`` -- so a bare string would otherwise ship
    # as a positional-params array of one character each.
    if params is None or isinstance(params, Mapping):
        return
    if isinstance(params, (list, tuple)):
        return
    raise ProtocolViolationError(
        INVALID_PARAMS,
        f"params must be an object or an array, got {_json_type_name(params)}",
    )


def _as_wire_params(params: Mapping[str, Any] | Sequence[Any]) -> Any:
    return dict(params) if isinstance(params, Mapping) else list(params)


@dataclass(frozen=True)
class Request:
    """A call that expects a reply."""

    id: int
    method: str
    params: Mapping[str, Any] | Sequence[Any] | None = None

    def __post_init__(self) -> None:
        _check_method(self.method)
        _check_params(self.params)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id, "method": self.method}
        # Omitted, not null: the spec permits an absent ``params`` but a
        # strict server rejects ``"params": null``.
        if self.params is not None:
            wire["params"] = _as_wire_params(self.params)
        return wire

    def encode(self) -> bytes:
        try:
            return json.dumps(self.to_wire()).encode("utf-8")
        except (TypeError, ValueError) as exc:
            # A non-serialisable argument is our bug, not the peer's, and it
            # has to say so -- ``TypeError: Object of type X is not JSON
            # serializable`` on its own does not name the request at fault.
            raise ProtocolViolationError(
                INTERNAL_ERROR,
                f"params for {self.method!r} could not be serialised to JSON: {exc}",
            ) from exc


@dataclass(frozen=True)
class Notification:
    """A call that must never be answered.

    The absence of ``id`` is the whole difference, and it is load-bearing in
    both directions: we never wait for a reply, and a peer that replies to one
    is reporting something we did not ask about (:class:`UnmatchedResponse`).
    """

    method: str
    params: Mapping[str, Any] | Sequence[Any] | None = None

    def __post_init__(self) -> None:
        _check_method(self.method)
        _check_params(self.params)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": self.method}
        if self.params is not None:
            wire["params"] = _as_wire_params(self.params)
        return wire

    def encode(self) -> bytes:
        try:
            return json.dumps(self.to_wire()).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProtocolViolationError(
                INTERNAL_ERROR,
                f"params for notification {self.method!r} could not be serialised to JSON: {exc}",
            ) from exc


@dataclass(frozen=True)
class Response:
    """A reply to a request we sent, or one we would send back.

    Exactly one of ``result``/``error`` is set; the parser guarantees it, so
    nothing downstream has to re-check. ``result`` may legitimately be ``None``
    -- which is why the parser keys off the *presence* of the member and not
    its truthiness.
    """

    id: str | int | None
    result: Any = None
    error: ErrorObject | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


@dataclass(frozen=True)
class InboundRequest:
    """A request *from* the peer.

    Legal JSON-RPC that bevel has no answers for yet -- m10c decides what to
    reply. Classifying it as malformed would be a lie, and replying to it from
    here would be m10c's vocabulary leaking backwards.
    """

    id: str | int | None
    method: str
    params: Any = None


@dataclass(frozen=True)
class InboundNotification:
    """A notification from the peer. Nothing to answer."""

    method: str
    params: Any = None


@dataclass(frozen=True)
class MalformedFrame:
    """A frame that arrived and could not be understood.

    A value, not an exception, because the reader loop must survive it: one
    bad line from a buggy server is not a reason to stop reading the pipe.
    ``raw`` keeps a clipped copy of the bytes -- without it the error says
    "not JSON" and the user still has no idea what the peer actually sent.
    """

    error: ErrorObject
    raw: str


@dataclass(frozen=True)
class UnmatchedResponse:
    """A reply whose ``id`` matches no outstanding request.

    Reported instead of dropped. A duplicate, a late answer to a cancelled
    call, or a peer inventing ids all look identical from here, and "the
    server answered something nobody asked for" is the only clue that tells you
    which.
    """

    id: str | int | None
    reason: str


Incoming = Response | MalformedFrame | InboundRequest | InboundNotification | UnmatchedResponse
"""Whatever one frame on the wire turned out to be.

:func:`parse_frame` never produces :class:`UnmatchedResponse` -- only the
channel can, because only the channel knows which ids it asked about.
"""


_JSON_TYPE_NAMES: Final = {
    type(None): "null",
    bool: "boolean",
    int: "number",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def _json_type_name(value: Any) -> str:
    """The JSON name of ``value``'s type, for error messages.

    Keyed on the exact type rather than ``isinstance`` chains so ``bool`` is
    reported as ``boolean`` and not as the number it subclasses to be.
    """
    return _JSON_TYPE_NAMES.get(type(value), type(value).__name__)


def _clip(text: str, limit: int = _RAW_CLIP) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _is_id(value: Any) -> bool:
    """Can this be a JSON-RPC id?

    Strings and integers only. ``null`` is excluded because bevel never issues
    one, so accepting it would let a peer's ``{"id": null}`` match a request
    that also has no id. Floats and booleans are excluded for the same reason:
    they cannot have come from our allocator, so a reply carrying one is not
    addressing us.
    """
    return isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))


def _read_error_object(raw: Any) -> tuple[ErrorObject | None, list[str]]:
    """Validate a peer's ``error`` member, collecting every defect at once."""
    if not isinstance(raw, dict):
        return None, [f"error must be an object, got {_json_type_name(raw)}"]

    problems: list[str] = []
    code = raw.get("code")
    message = raw.get("message")
    # ``bool`` is excluded because it subclasses ``int`` in Python, and a peer
    # that answers with ``{"code": true}`` must not get an error object whose
    # code silently reads as 1.
    if not isinstance(code, int) or isinstance(code, bool):
        problems.append(f"error.code must be an integer, got {_json_type_name(code)}")
    if not isinstance(message, str) or not message:
        problems.append(f"error.message must be a non-empty string, got {_json_type_name(message)}")
    if problems:
        return None, problems
    # The two checks above have already proved these types; mypy cannot follow
    # that conclusion through the ``problems`` list, so it is asserted here.
    return ErrorObject(
        code=cast("int", code), message=cast("str", message), data=raw.get("data")
    ), []


def _parse_response(obj: dict[str, Any], raw: str) -> Response | MalformedFrame:
    problems: list[str] = []
    if obj.get("jsonrpc") != JSONRPC_VERSION:
        problems.append(
            f'jsonrpc must be "{JSONRPC_VERSION}", got {_json_type_name(obj.get("jsonrpc"))}'
        )
    request_id = obj.get("id")
    if not _is_id(request_id):
        problems.append(f"id must be a string or an integer, got {_json_type_name(request_id)}")

    has_result = "result" in obj
    has_error = "error" in obj
    if has_result and has_error:
        problems.append("a response carries either result or error, not both")
    elif not has_result and not has_error:
        problems.append("a response carries one of result or error, and this carries neither")

    error: ErrorObject | None = None
    if has_error:
        error, error_problems = _read_error_object(obj["error"])
        problems.extend(error_problems)

    if problems:
        # Every defect in one message, the same rule the tool-argument
        # validator follows: whoever fixes this gets the whole list in one
        # pass instead of discovering them one re-ask at a time.
        return MalformedFrame(ErrorObject.invalid_request("; ".join(problems)), raw=raw)
    # Subscripting rather than re-reading through ``.get``: the checks above
    # have already proved these members exist and have the right types.
    return Response(id=obj["id"], result=obj["result"] if has_result else None, error=error)


def _parse_call(obj: dict[str, Any], raw: str) -> Incoming:
    problems: list[str] = []
    if obj.get("jsonrpc") != JSONRPC_VERSION:
        problems.append(
            f'jsonrpc must be "{JSONRPC_VERSION}", got {_json_type_name(obj.get("jsonrpc"))}'
        )
    method = obj.get("method")
    if not isinstance(method, str) or not method:
        problems.append(f"method must be a non-empty string, got {_json_type_name(method)}")
    params = obj.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        problems.append(f"params must be an object or an array, got {_json_type_name(params)}")
    has_id = "id" in obj
    if has_id and not _is_id(obj["id"]):
        problems.append(f"id must be a string or an integer, got {_json_type_name(obj['id'])}")

    if problems:
        return MalformedFrame(ErrorObject.invalid_request("; ".join(problems)), raw=raw)
    # Subscripted re-reads: the checks above have already proved ``method`` is
    # a non-empty string and, when present, ``id`` is a string or integer.
    if has_id:
        return InboundRequest(id=obj["id"], method=obj["method"], params=params)
    return InboundNotification(method=obj["method"], params=params)


def parse_frame(data: bytes | str) -> Incoming:
    """Decode one frame. Total: bytes in, an :data:`Incoming` out, never a raise.

    The totality is the contract. A ``ValueError`` escaping here becomes an
    exception in whatever loop is reading the pipe, and a dead reader is a
    client that hangs waiting for replies it is no longer reading. Every
    failure instead comes back as a :class:`MalformedFrame` carrying the spec
    code that names what was wrong.
    """
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            return MalformedFrame(
                ErrorObject.parse_error(f"frame is not UTF-8 text: {exc}"),
                raw=_clip(repr(data)),
            )
    else:
        text = data

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        return MalformedFrame(ErrorObject.parse_error(f"frame is not JSON: {exc}"), raw=_clip(text))

    if not isinstance(obj, dict):
        name = _json_type_name(obj)
        detail = (
            "expected a JSON-RPC object, got an array; bevel does not speak top-level batches"
            if name == "array"
            else f"expected a JSON-RPC object, got {name}"
        )
        return MalformedFrame(ErrorObject.invalid_request(detail), raw=_clip(text))

    if "method" in obj:
        return _parse_call(obj, _clip(text))
    return _parse_response(obj, _clip(text))


@runtime_checkable
class DuplexStream(Protocol):
    """The only thing this module asks of the outside world: framed bytes both ways.

    Deliberately two methods and no more. It is not ``asyncio``'s
    ``StreamReader`` -- that one has eleven methods, speaks in chunks rather
    than frames, and cannot be satisfied by an object holding a list of
    ``bytes``. A test implements this with a queue and a list; m10b implements
    it with a pipe and the newline framing that belongs to it.

    Framing is the implementation's job, not ours. ``read_frame`` returns one
    complete message or ``None`` at end of stream; ``write_frame`` accepts one
    complete message and is responsible for whatever delimiter the transport
    needs. Keeping delimiters out of here is what lets stdio (newline-delimited)
    and streamable-HTTP (JSON bodies) share this pairing layer unchanged.
    """

    async def read_frame(self) -> bytes | None:
        """Return the next complete frame, or ``None`` once the peer is done."""
        ...

    async def write_frame(self, payload: bytes) -> None:
        """Send one complete frame."""
        ...


class PendingRequest:
    """One request waiting for the reply that carries its id.

    Handed back by :meth:`JsonRpcChannel.request` so a caller can send several
    requests before waiting on any of them. ``id`` and ``method`` are public
    so a caller can tell which of its own requests a handle belongs to; the
    future is private so that only the channel can settle it, which is what
    keeps "exactly one caller gets each response" true.
    """

    __slots__ = ("_future", "id", "method")

    def __init__(self, id: int, method: str, future: asyncio.Future[Any]) -> None:
        self.id = id
        self.method = method
        self._future = future

    @property
    def done(self) -> bool:
        return self._future.done()

    async def result(self) -> Any:
        """Await the reply.

        Re-raises whatever the channel decided the answer was:
        :class:`RemoteError` for a peer error, :class:`ProtocolViolationError` when
        the frame meant for this call was unreadable, :class:`StreamClosedError`
        when the pipe died. A caller that never awaits this is still not left
        hanging -- closing the channel fails it anyway.
        """
        return await self._future

    def _resolve(self, value: Any) -> None:
        if not self._future.done():
            self._future.set_result(value)

    def _fail(self, error: BaseException) -> None:
        if not self._future.done():
            self._future.set_exception(error)

    def _discard(self) -> None:
        """Mark this future's exception as retrieved.

        A caller that has stopped caring -- it cancelled, or it already raised
        past us -- must not leave the exception sitting unread. asyncio prints
        "Future exception was never retrieved" when such a future is collected,
        which turns a correctly handled cancellation into spurious noise in the
        TUI's log.
        """
        if self._future.done() and not self._future.cancelled():
            self._future.exception()


@dataclass
class IdAllocator:
    """Hands out request ids, monotonically, and refuses to wrap.

    Not frozen: counting is its whole job. Ids start at 1 because 0 is a
    legitimate id a peer might invent, and matching one we never issued against
    one we did would be exactly the confusion this class exists to prevent.
    """

    limit: int = MAX_SAFE_INTEGER
    issued: int = field(default=0, init=False)

    def allocate(self) -> int:
        value = self.issued + 1
        if value > self.limit:
            raise IdExhaustedError(
                f"request ids are exhausted at {self.limit}; this allocator has issued "
                f"{self.issued} ids and cannot issue another without reusing one"
            )
        self.issued += 1
        return value


class JsonRpcChannel:
    """Pairs requests to responses over a :class:`DuplexStream`.

    One channel per connection. It owns the id counter, the outstanding
    requests, and the record of everything that arrived that could not be
    matched. It does not own the read loop: the transport calls
    :meth:`receive` (or uses :meth:`call`, which pumps it) so that whoever
    owns the pipe also owns when reads happen.

    Responses are matched by id alone, so a peer may answer out of order and
    each caller still gets its own answer.
    """

    def __init__(self, stream: DuplexStream) -> None:
        self._stream = stream
        self._allocator = IdAllocator()
        self._pending: dict[str | int, PendingRequest] = {}
        self._reports: list[MalformedFrame | UnmatchedResponse] = []
        self._closed = False

    @property
    def pending_ids(self) -> tuple[int, ...]:
        """Ids still waiting for a reply, in the order they were issued."""
        return tuple(pending.id for pending in self._pending.values())

    @property
    def reports(self) -> tuple[MalformedFrame | UnmatchedResponse, ...]:
        """Everything that arrived and could not be delivered.

        Kept rather than logged-and-forgotten so a TUI (m10d) can show the
        user "the peer sent three frames we could not parse" instead of the
        session merely feeling slow.
        """
        return tuple(self._reports)

    async def request(
        self, method: str, params: Mapping[str, Any] | Sequence[Any] | None = None
    ) -> PendingRequest:
        """Send a request without waiting for its reply.

        The write happens before this returns, so the peer has the request by
        the time the caller holds a :class:`PendingRequest` -- there is no
        window in which a reply could arrive for a request that was never sent.
        """
        if self._closed:
            raise StreamClosedError("the stream is closed; no request can be sent")
        outgoing = Request(id=self._allocator.allocate(), method=method, params=params)
        pending = PendingRequest(
            id=outgoing.id, method=method, future=asyncio.get_running_loop().create_future()
        )
        self._pending[outgoing.id] = pending
        try:
            await self._stream.write_frame(outgoing.encode())
        except BaseException:
            # A failed write must not leave a phantom request that a later
            # id collision could satisfy.
            self._pending.pop(outgoing.id, None)
            raise
        return pending

    async def call(
        self, method: str, params: Mapping[str, Any] | Sequence[Any] | None = None
    ) -> Any:
        """Send a request and await its result.

        Raises :class:`RemoteError` when the peer answers with an error,
        :class:`ProtocolViolationError` when the frame that arrives is unreadable,
        and :class:`StreamClosedError` when the pipe ends first. In all three
        cases the request stops being outstanding -- a caller that has been
        told "no" is never left registered.
        """
        pending = await self.request(method, params)
        try:
            while not pending.done:
                await self.receive()
            return await pending.result()
        except BaseException:
            # Covers cancellation too: a Ctrl+C mid-call must not leave the id
            # registered, or the next reply for it is swallowed by a caller
            # that has gone away.
            self._pending.pop(pending.id, None)
            pending._discard()
            raise

    async def notify(
        self, method: str, params: Mapping[str, Any] | Sequence[Any] | None = None
    ) -> None:
        """Send a notification. Returns once written; waits for nothing.

        No id is consumed, so a notification between two calls does not shift
        the second call's id.
        """
        if self._closed:
            raise StreamClosedError("the stream is closed; no notification can be sent")
        await self._stream.write_frame(Notification(method=method, params=params).encode())

    async def receive(self) -> Incoming:
        """Read one frame, deliver it if it matches, and hand it back.

        Returns whatever the frame was -- including the things the caller has
        no use for -- because the transport's read loop needs to see inbound
        requests and notifications too. Anything undeliverable is recorded in
        :attr:`reports` on the way out.
        """
        if self._closed:
            raise StreamClosedError("the stream is closed")
        data = await self._stream.read_frame()
        if data is None:
            raise self._shutdown()
        return self._dispatch(parse_frame(data))

    def _dispatch(self, frame: Incoming) -> Incoming:
        if isinstance(frame, Response):
            return self._deliver(frame)
        if isinstance(frame, MalformedFrame):
            self._reports.append(frame)
            self._fail_lone_pending(frame)
        return frame

    def _deliver(self, frame: Response) -> Incoming:
        key = frame.id
        pending = self._pending.pop(key, None) if key is not None else None
        if pending is None:
            waiting = ", ".join(str(i) for i in self.pending_ids) or "none"
            unmatched = UnmatchedResponse(
                id=frame.id,
                reason=(f"no request is waiting for id {frame.id!r} (outstanding ids: {waiting})"),
            )
            self._reports.append(unmatched)
            return unmatched
        if frame.error is not None:
            pending._fail(
                RemoteError(
                    frame.error.code,
                    frame.error.message,
                    data=frame.error.data,
                    request_id=frame.id,
                    method=pending.method,
                )
            )
        else:
            pending._resolve(frame.result)
        return frame

    def _fail_lone_pending(self, frame: MalformedFrame) -> None:
        """Attribute an unreadable frame when attribution is actually possible.

        A frame we cannot parse carries no id, so with several requests in
        flight there is no honest way to say which one it answered -- failing a
        guess would punish the wrong caller. With exactly one outstanding, the
        answer is unambiguous and the alternative is that caller blocking
        forever on a reply that already arrived broken, so it fails. Either
        way the frame lands in :attr:`reports`; nothing is swallowed.
        """
        if len(self._pending) != 1:
            return
        pending = next(iter(self._pending.values()))
        self._pending.pop(pending.id, None)
        pending._fail(
            ProtocolViolationError(frame.error.code, frame.error.message, frame.error.data)
        )

    def _shutdown(self) -> StreamClosedError:
        self._closed = True
        waiting = list(self._pending.values())
        self._pending.clear()
        names = ", ".join(pending.method for pending in waiting) or "nothing"
        message = (
            f"the stream closed with {len(waiting)} request(s) outstanding ({names}); "
            "their replies can no longer arrive"
        )
        # Every outstanding caller is failed, not just the one that happened to
        # be reading. Failing only the reader is how the others hang.
        for pending in waiting:
            pending._fail(StreamClosedError(message))
        raise StreamClosedError(message)

    def _closed_error(self) -> StreamClosedError:
        return StreamClosedError("the stream is closed")

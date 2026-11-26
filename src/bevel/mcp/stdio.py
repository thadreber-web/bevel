"""Spawn an MCP server by argv and speak newline-delimited JSON-RPC to it.

This is the m10b transport: one child process, its stdin and stdout, and the
framing that turns a byte stream into the frames
:class:`~bevel.mcp.jsonrpc.JsonRpcChannel` pairs. It satisfies
:class:`~bevel.mcp.jsonrpc.DuplexStream` and nothing wider — two methods, no
third, because the pairing layer deliberately knows nothing about pipes.

**The reader is a background task, and here is why.** A server that sends three
notifications and then a response is perfectly well behaved. A client that
touches the pipe only when it wants an answer has to pull those three
notifications off it before it reaches the reply — and a server whose
notifications exceed the ~64 KiB pipe buffer blocks in ``write`` while the
client blocks in ``read`` waiting for a response queued behind them. Two
processes, each waiting on the other, forever. So a task started by
:meth:`spawn` calls ``read`` on stdout continuously, whether or not anyone is
waiting, into a buffer that :meth:`read_frame` carves frames out of. The OS pipe
is therefore always draining, which is what makes the
three-notifications-then-a-response case unreachable rather than merely
unlikely. stderr gets the same treatment for the same reason: a server that
logs, and is never drained, stops being able to log, and most servers treat a
failed ``write`` as fatal.

**What this module is not.** It speaks ``initialize`` and ``tools/list`` on the
wire and stops there. It does not know what a ``ToolSpec`` is, does not
namespace remote tool names, and does not turn content blocks into a
``ToolResult`` — that translation is m10c, and the absence of any import from
``core/tools/`` here is the boundary holding.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from bevel.core._process import TreeKiller, close_process, uses_new_session
from bevel.mcp.jsonrpc import JsonRpcChannel, RemoteError, StreamClosedError

#: Protocol versions this client will accept from a server, newest first.
SUPPORTED_PROTOCOL_VERSIONS: Final[tuple[str, ...]] = ("2025-06-18", "2025-03-26", "2024-11-05")

#: How long to wait for the child to be created.
DEFAULT_SPAWN_TIMEOUT: Final[float] = 15.0
#: How long to wait for the ``initialize`` / ``tools/list`` exchange.
DEFAULT_HANDSHAKE_TIMEOUT: Final[float] = 15.0
#: How long to wait for one request's reply.
DEFAULT_CALL_TIMEOUT: Final[float] = 60.0

#: A child that dies inside this window never got the chance to be spoken to, so
#: it is a spawn failure rather than a handshake one. Comfortably above the
#: ~0.3 s a healthy CPython child needs to start on a loaded runner.
LIVENESS_WINDOW: Final[float] = 1.5

#: How long a killed child gets to actually be reaped before we stop waiting.
_REAP_TIMEOUT: Final[float] = 5.0

_CHUNK_BYTES: Final[int] = 65_536
_NEWLINE: Final[bytes] = b"\n"


class McpTransportError(Exception):
    """Base for everything the stdio transport can get wrong.

    One base so the TUI (m10d) can guard a whole session with a single
    ``except`` and still branch on the stage when it wants to say something more
    useful than "it failed".
    """


class McpSpawnError(McpTransportError):
    """The child never became a server: it could not be created, or died at once.

    The message names ``argv`` and, when there is one, the exit code. "It timed
    out" would send someone looking for a slow server when the answer is that
    the program does not run.
    """


class McpTimeoutError(McpTransportError):
    """One stage of the session ran out of time.

    The stage is always in the message. A spawn, handshake and call timeout are
    three different bugs in three different files, and an undifferentiated "MCP
    timed out" is how a CI hang stays undiagnosed.
    """


class McpHandshakeError(McpTransportError):
    """The server answered, but not in a way bevel can speak.

    A refused protocol version, an ``initialize`` result with no shape, a
    ``tools/list`` with no tools. Each means every later request would be
    interpreted under rules neither end agreed to, so the session ends here.
    """


class StdioTransport:
    """A child process spoken to as an MCP server.

    Lifecycle: :meth:`spawn` (or :meth:`start`, which spawns), then
    :meth:`call` and :meth:`notify`, then :meth:`close`. Teardown kills the whole
    process tree through :class:`~bevel.core._process.TreeKiller` on every exit
    path, including the ones taken by an exception or a cancellation — a server
    that outlives its client is how a CI runner accumulates orphans.
    """

    #: Bytes of stderr kept for error messages. Bounded because the server, not
    #: us, decides how much it writes, and an unbounded buffer is a memory leak
    #: with a friendly name.
    STDERR_CAP: int = 65_536

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        spawn_timeout: float = DEFAULT_SPAWN_TIMEOUT,
        handshake_timeout: float = DEFAULT_HANDSHAKE_TIMEOUT,
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> None:
        if not argv:
            raise ValueError("argv must name the program to run")
        self._argv: tuple[str, ...] = tuple(str(item) for item in argv)
        self._cwd = cwd
        self._env = dict(env) if env is not None else None
        self._spawn_timeout = spawn_timeout
        self._handshake_timeout = handshake_timeout
        self._call_timeout = call_timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._killer: TreeKiller | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = bytearray()
        self._buffer = bytearray()
        self._data = asyncio.Event()
        # Two concurrent call()s mean two concurrent read_frame()s, because
        # JsonRpcChannel.call pumps receive() itself. Without this lock both can
        # see the same newline and hand back the same frame, which dispatches one
        # reply twice. Holding it across the wait is deliberate: one reader waits
        # for bytes, the others queue behind it, and each frame goes to exactly
        # one of them. A cancelled reader releases it, so a timed-out call cannot
        # wedge the ones behind it.
        self._read_lock = asyncio.Lock()
        self._eof = False
        self._closed = False
        self._server_info: dict[str, Any] = {}
        self._protocol_version: str | None = None
        self._channel = JsonRpcChannel(self)

    # --- the two methods DuplexStream asks for ------------------------------

    async def read_frame(self) -> bytes | None:
        """Return the next complete line without its delimiter, or ``None`` at EOF.

        One ``read`` is not one frame. The kernel hands back whatever has
        arrived, so a frame can straddle two reads and two frames can arrive in
        one; the bytes left over after a delimiter stay in the buffer until their
        own delimiter shows up. A final frame that reaches EOF without a
        delimiter is still returned rather than dropped — dropping it would lose
        a server's last words, and the channel is better placed to judge whether
        it parses.
        """
        if self._proc is None:
            raise StreamClosedError("the transport is not spawned; there is no stdout to read")
        async with self._read_lock:
            while True:
                self._data.clear()
                end = self._buffer.find(_NEWLINE)
                if end >= 0:
                    frame = bytes(self._buffer[:end])
                    del self._buffer[: end + 1]
                    return frame
                if self._eof:
                    if not self._buffer:
                        return None
                    frame = bytes(self._buffer)
                    self._buffer.clear()
                    return frame
                await self._data.wait()

    async def write_frame(self, payload: bytes) -> None:
        """Send one frame, appending the newline that terminates it.

        The delimiter belongs to the transport, which is why the channel hands
        over bare bytes: stdio's frame is a line, and streamable-HTTP's (m10e) is
        a JSON body.
        """
        if self._closed:
            raise StreamClosedError("the transport is closed; nothing can be written")
        if self._proc is None or self._proc.stdin is None:
            raise StreamClosedError("the transport is not spawned; there is no stdin to write")
        self._proc.stdin.write(payload + _NEWLINE)
        try:
            await self._proc.stdin.drain()
        except OSError as exc:
            # Includes BrokenPipeError and ConnectionResetError: the server is
            # gone, and a frame nobody received is a closed stream.
            raise StreamClosedError(
                f"the MCP server {self._argv[0]!r} closed its stdin before accepting a frame"
            ) from exc

    # --- lifecycle ----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        """The server's pid, or ``None`` once it has been torn down."""
        return self._proc.pid if self._proc is not None else None

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
    def stderr_tail(self) -> str:
        """The last :attr:`STDERR_CAP` bytes of stderr, decoded leniently.

        This is the text that turns "the server hung" into a diagnosis, so every
        transport error quotes it.
        """
        return self._stderr.decode("utf-8", errors="replace")

    async def spawn(self) -> None:
        """Create the child, start draining both pipes, and confirm it is alive.

        Split from :meth:`start` so a test can drive the framing by hand — write
        a raw line, read one back — without a handshake in the way.
        """
        if self._proc is not None:
            return
        kwargs: dict[str, Any] = {
            "stdin": asyncio.subprocess.PIPE,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "cwd": str(self._cwd) if self._cwd is not None else None,
            "env": self._env,
        }
        if uses_new_session():
            # Popen raises ValueError for this on Windows; there the Job Object
            # inside TreeKiller is what reaches the descendants.
            kwargs["start_new_session"] = True

        program = self._argv[0]
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(*self._argv, **kwargs),
                self._spawn_timeout,
            )
        except TimeoutError as exc:
            raise McpTimeoutError(
                f"spawn timed out after {self._spawn_timeout:.1f}s: the operating system never "
                f"handed back a process for {program!r} ({self._describe()}). Nothing was "
                f"written to it, so this is not a server refusing to talk."
            ) from exc
        except (FileNotFoundError, PermissionError) as exc:
            raise McpSpawnError(
                f"spawn failed: {program!r} could not be started ({exc}). Check the command in "
                f"{self._describe()} and that it is installed and executable."
            ) from exc
        except OSError as exc:
            raise McpSpawnError(
                f"spawn failed: the operating system refused to start {program!r} ({exc}) for "
                f"{self._describe()}."
            ) from exc

        self._closed = False
        self._proc = proc
        self._killer = TreeKiller(proc.pid)
        self._stdout_task = asyncio.create_task(self._pump_stdout())
        self._stderr_task = asyncio.create_task(self._pump_stderr())
        await self._confirm_alive()

    async def start(self) -> list[dict[str, Any]]:
        """Spawn, negotiate the protocol version, and return the advertised tools.

        ``initialize`` → ``notifications/initialized`` → ``tools/list``, and a
        peer that will not negotiate is rejected here rather than discovered
        three calls later. The notification goes before ``tools/list`` because a
        strict server refuses any request that arrives first.
        """
        await self.spawn()
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
        broken pipe — and :class:`McpTimeoutError` naming the call stage when the
        reply does not come.
        """
        if self._closed:
            raise StreamClosedError(f"the transport is closed; {method!r} was not sent")
        budget = self._call_timeout if timeout is None else timeout
        try:
            return await asyncio.wait_for(self._channel.call(method, params), budget)
        except TimeoutError as exc:
            raise McpTimeoutError(
                f"call timed out after {budget:.1f}s: the MCP server {self._argv[0]!r} "
                f"({self._describe()}) never answered {method!r}. {self._stderr_clause()}"
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

    async def kill(self) -> None:
        """Terminate the server's whole process tree now, without tearing down.

        Separate from :meth:`close` because a test has to make the pipe die while
        requests are in flight and watch what the channel does with them. The
        pumps are deliberately left running: EOF is the event that fails the
        outstanding requests, and stopping them first would leave those callers
        waiting on a pipe nobody is reading.
        """
        if self._killer is not None:
            self._killer.kill()

    async def close(self) -> None:
        """Kill the tree, reap the child, and release everything. Safe twice."""
        self._closed = True
        stdout_task, stderr_task = self._stdout_task, self._stderr_task
        self._stdout_task = self._stderr_task = None
        proc, killer = self._proc, self._killer
        self._proc = self._killer = None

        if proc is not None and killer is not None:
            # A child we already reaped gets its handle released; anything else
            # is killed, Job Object and all.
            if proc.returncode is None:
                killer.kill()
            else:
                killer.close()

        # Let both pumps reach EOF before anything is cancelled. That is the path
        # on which the channel fails the outstanding requests; cancelling them
        # first is how a caller ends up waiting for a reply that will never be
        # dispatched.
        for task in (stdout_task, stderr_task):
            if task is None:
                continue
            try:
                await asyncio.wait_for(task, _REAP_TIMEOUT)
            except (TimeoutError, StreamClosedError):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        if proc is not None:
            await self._reap(proc, killer)

    # --- internals ----------------------------------------------------------

    async def _reap(self, proc: asyncio.subprocess.Process, killer: TreeKiller | None) -> None:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT)
            except TimeoutError:
                # The tree is already killed; the kernel is just slow to say so.
                if killer is not None:
                    killer.kill()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(proc.wait(), _REAP_TIMEOUT)
        close_process(proc)

    async def _confirm_alive(self) -> None:
        """Fail now if the child died before it could be spoken to.

        ``create_subprocess_exec`` returns in milliseconds whether or not the
        program works, and a stdio MCP server sends nothing until it is asked, so
        "spawn returned" carries no information on its own. Racing a short wait
        against the child's exit is what separates "that program does not run"
        from "that server will not answer" — and it has to happen before the
        first write, because a server that dies *after* reading a request is an
        EOF, which is a different failure with a different message.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            await asyncio.wait_for(proc.wait(), LIVENESS_WINDOW)
        except TimeoutError:
            return
        code = proc.returncode
        await self.close()
        raise McpSpawnError(
            f"spawn failed: {self._argv[0]!r} ({self._describe()}) exited with code {code} "
            f"before it could be spoken to. It never received a request, so this is not a "
            f"handshake timeout. {self._stderr_clause()}"
        )

    async def _handshake(self, method: str, params: Mapping[str, Any], waiting_for: str) -> Any:
        """One request during :meth:`start`, under the handshake budget."""
        try:
            return await asyncio.wait_for(
                self._channel.call(method, params), self._handshake_timeout
            )
        except TimeoutError as exc:
            raise McpTimeoutError(
                f"handshake timed out after {self._handshake_timeout:.1f}s: the MCP server "
                f"{self._argv[0]!r} ({self._describe()}) was running but never answered "
                f"{method!r}, while waiting for {waiting_for}. {self._stderr_clause()}"
            ) from exc
        except RemoteError as exc:
            raise McpHandshakeError(
                f"handshake failed: the MCP server {self._argv[0]!r} ({self._describe()}) "
                f"rejected {method!r} with code {exc.code}: {exc.message}. "
                f"{self._stderr_clause()}"
            ) from exc

    def _agree_on_version(self, result: Any) -> None:
        """Accept a version we support; refuse a peer that will not negotiate.

        A server that omits ``protocolVersion``, or names one we do not speak, is
        not a server to guess at: every later request would be interpreted under
        rules neither side agreed to.
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

    async def _pump_stdout(self) -> None:
        """Drain stdout continuously so the pipe is never what blocks the server."""
        stream = self._proc.stdout if self._proc is not None else None
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(_CHUNK_BYTES)
                if not chunk:
                    return
                self._buffer.extend(chunk)
                self._data.set()
        finally:
            # Whoever is parked in read_frame must wake up and see the end.
            self._eof = True
            self._data.set()

    async def _pump_stderr(self) -> None:
        """Keep stderr moving so a chatty server is never blocked by us.

        Reads past the cap on purpose. Keeping only the tail is a memory
        decision; stopping reading would be a deadlock, and the difference
        between those two sentences is the whole point of this method.
        """
        stream = self._proc.stderr if self._proc is not None else None
        if stream is None:
            return
        while True:
            chunk = await stream.read(_CHUNK_BYTES)
            if not chunk:
                return
            self._stderr.extend(chunk)
            if len(self._stderr) > self.STDERR_CAP:
                del self._stderr[: len(self._stderr) - self.STDERR_CAP]

    def _describe(self) -> str:
        # The command line as the operating system would parse it, so the string
        # in the error is the string you can paste into a shell and reproduce.
        return subprocess.list2cmdline(self._argv)

    def _stderr_clause(self) -> str:
        tail = self.stderr_tail.strip()
        return (
            f"Last stderr from the server: {tail!r}"
            if tail
            else "The server wrote nothing to stderr."
        )


def _tool_list(result: Any, described_argv: str) -> list[dict[str, Any]]:
    """Pull ``tools`` out of a ``tools/list`` result, or say why there is none.

    Validates rather than casts: a server that answers ``{"tools": "yes"}`` is a
    server that must be told so, not one whose nonsense is laundered into a
    comprehension that fails somewhere less obvious.
    """
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        raise McpHandshakeError(
            f"handshake failed: tools/list returned {result!r:.200} from {described_argv}, "
            f"which has no 'tools' array."
        )
    return [tool for tool in result["tools"] if isinstance(tool, dict)]


def _client_version() -> str:
    from bevel import __version__  # noqa: PLC0415 - a module-scope import would cycle bevel -> mcp

    return __version__

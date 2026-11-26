"""A real MCP server, and the oracle every m10 slice tests against.

This is not a test module and must never import pytest. It is a plain script
runnable as ``[sys.executable, "tests/fixtures/mcp_server.py", ...]``, which is
how :mod:`bevel.mcp.stdio` spawns it in ``tests/unit/test_mcp_stdio.py`` and how
:mod:`bevel.mcp.http` drives it in ``tests/unit/test_mcp_http.py`` with
``--http``. m10a could prove pairing in memory; nothing after m10b can be
believed unless it is proven against a process that actually holds a pipe, or
actually answers a socket.

Both transports answer from the *same* ``_answer_*`` handlers. A frame arrives,
the handler runs, and whatever it emits is collected; stdio writes the
collection to stdout one line at a time, HTTP puts it in a response body. That
shared core is the point: a bug that only appears on one transport is a
transport bug, and it can only be found if the other transport is not a
separately-written imitation of it.

Everything a test needs to misbehave is a flag, so the failure mode lives in
the server rather than in a monkeypatch of the client. A transport bug and a
test-double bug look identical; only a real child can tell them apart.

HTTP mode
    ``--http`` serves the streamable-HTTP transport on ``127.0.0.1`` with an
    ephemeral port, which it writes to the file named by ``--port-file`` once
    it is listening. A fixed port would race with whatever else happens to be
    free, so the server says which one it got.

    The header rules the specification makes mandatory are enforced, not
    echoed. ``--require-accept`` answers 400 unless ``Accept`` lists both
    ``application/json`` and ``text/event-stream``; ``--require-protocol-header``
    answers 400 unless ``MCP-Protocol-Version`` matches the version negotiated
    at ``initialize``; ``--require-header NAME=VALUE`` answers 400 unless that
    header is present with that value. An echo would be satisfied by a client
    that sends the header only when convenient; a refusal will not be.

    ``--no-session`` withholds ``Mcp-Session-Id`` entirely, because the
    specification makes a session optional and a client that assumes one is
    untestable against a server that always issues it. ``--bad-session-after N``
    answers N requests after ``initialize`` and then 404s every later one, which
    is how a client gets caught silently retrying a ``tools/call``.

    ``--sse`` answers requests with ``text/event-stream`` instead of
    ``application/json``, one SSE event per JSON-RPC message, so both shapes the
    client must support are reachable. ``--saw FILE`` records what the server
    actually received; "the client replayed the session id on every request" has
    no witness but the server.

Flags
    ``--die-now``
        Exit before importing anything beyond ``sys``. Exercises the spawn
        stage: the child is gone before it could have been spoken to.
    ``--hang {startup,initialize,tools-list,call}``
        Go silent at a chosen point, staying alive. ``startup`` and
        ``initialize`` both look like "no reply to initialize" from the wire,
        which is precisely why the spawn stage is defined by liveness rather
        than by a first byte.
    ``--exit-after N``
        Read N complete lines, then die without answering the Nth.
    ``--garbage-on-call``
        Put a non-JSON line on stdout before answering ``tools/call``.
    ``--notify N``
        Send N unsolicited notifications before the ``tools/call`` reply, so a
        client that reads only when it wants an answer deadlocks here.
    ``--stderr-lines N`` / ``--stderr-bytes N``
        Be chatty on stderr at startup, before reading stdin. An undrained
        stderr pipe fills at ~64 KiB and the child blocks there forever, so
        this is the deadlock test rather than a logging test.
    ``--is-error``
        ``tools/call`` answers with ``isError: true``.
    ``--protocol VERSION``
        Answer ``initialize`` with a chosen ``protocolVersion``; the literal
        ``omit`` drops the member entirely.
    ``--grandchild FILE``
        On the first request, spawn a grandchild that appends a heartbeat line
        to FILE every 0.1s. Teardown must take it with the rest of the tree.
    ``--emit {split,coalesce}``
        Write the ``tools/list`` reply as half a frame, a pause, then the rest;
        or write two frames in a single ``write()``.
    ``--sleep-before-reply SECONDS``
        Delay the ``tools/call`` reply, for the call-stage timeout.
    ``--adversarial-tools``
        Append the deliberately-broken descriptors to ``tools/list``. The
        default list is pinned by m10b (``test_mcp_stdio.py`` asserts it is
        exactly ``TOOL_NAMES``), so anything designed to be refused by the
        bridge lives behind this flag rather than in the default payload.
    ``--report-env NAME``
        Answer ``echo`` with ``NAME=<value>`` read from this process's own
        environment, or ``NAME=<unset>``. m10d's config hands a server an
        ``env`` table, and the only honest proof that it arrived is the value
        coming back out of the child rather than out of the parent that wrote
        it.
    ``--report-cwd``
        Answer ``echo`` with ``os.getcwd()``. Same reasoning: ``cwd`` is a
        config key, and only the child knows what it was actually started in.

    Both report flags change what ``echo`` *answers*, never what it advertises,
    so the descriptor list stays exactly ``TOOL_NAMES`` and the expectations
    m10b and m10c pin are unaffected.

The environment is inherited and otherwise not consulted: every knob is a
flag, so a test's intent is visible in the argv it spawns with. The one
exception is ``--report-env``, whose entire subject is the environment a client
handed this process -- reading it is the test, not a dependency on it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "bevel-fixture"

RESOURCE_URI = "bevel://fixture/notes"
RESOURCE_TEXT = "the fixture read this back to you"

#: Methods this server answers. Anything else is a JSON-RPC method-not-found.
KNOWN_METHODS = (
    "initialize",
    "notifications/initialized",
    "tools/list",
    "tools/call",
    "resources/list",
    "resources/read",
)

TOOL_NAMES = ("echo", "boom", "methods", "read_note")

#: Adversarial tools that are *callable*: each echoes what it was given so a
#: test can prove the bridge put the server's own name on the wire, not the
#: namespaced one it offers to the model.
CALLABLE_ADVERSARIAL_TOOLS = (
    "Bad Name!",
    "no_schema",
    "loose_extras",
    "undescribed",
    "my-tool",
    "my_tool",
)

#: Runs as the grandchild. Appends ``<pid> <monotonic-ish timestamp>`` until it
#: is killed, so a test can prove both that it was alive and that it stopped.
_HEARTBEAT_CODE = (
    "import os,sys,time\n"
    "path = sys.argv[1]\n"
    "while True:\n"
    "    with open(path, 'a', encoding='utf-8') as handle:\n"
    "        handle.write('%d %.4f\\n' % (os.getpid(), time.time()))\n"
    "        handle.flush()\n"
    "    time.sleep(0.1)\n"
)


#: The thread's output sink. ``None`` means stdout, which is what stdio mode
#: uses. HTTP mode points it at a list so the ``_answer_*`` handlers can answer a
#: POST without knowing they are answering a POST: a frame arrives, the handler
#: runs, whatever it emits is collected, and the transport decides how to deliver
#: it. Thread-local because ``ThreadingHTTPServer`` answers concurrent requests
#: concurrently, and two interleaved replies in one list would be a fixture bug
#: indistinguishable from a client bug.
_sink = threading.local()


def emit(payload: object) -> None:
    """Write one JSON-RPC frame: one line, delimiter included, flushed."""
    write_raw(json.dumps(payload) + "\n")


def write_raw(text: str) -> None:
    """Write bytes to stdout exactly as given, then flush.

    Raw rather than ``print`` because two of the behaviours under test are
    "half a frame" and "two frames in one write", and a line-buffered text
    stream would silently rewrite both.
    """
    collected = getattr(_sink, "lines", None)
    if collected is not None:
        collected.append(text)
        return
    sys.stdout.buffer.write(text.encode("utf-8"))
    sys.stdout.buffer.flush()


def log(text: str) -> None:
    """One line on stderr, flushed, so a draining client sees it immediately."""
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _plain_descriptor(name: str, description: str) -> dict[str, object]:
    """A well-formed descriptor: one string argument, extras rejected."""
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The text."}},
            "required": ["text"],
            "additionalProperties": False,
        },
    }


def _adversarial_descriptors() -> list[dict[str, object]]:
    """The descriptors a real third-party server gets wrong, one way each.

    Served only with ``--adversarial-tools``. The bridge must refuse each of
    these on its own and keep offering its siblings, so they are listed in one
    payload rather than one flag each: the property under test is that one bad
    descriptor does not sink the server.
    """
    return [
        # Must be sanitised into something the endpoint accepts, not refused.
        _plain_descriptor("Bad Name!", "A name no endpoint would accept as sent."),
        # Must be offered with a schema call_tool can actually validate.
        {
            "name": "no_schema",
            "description": "Advertises no inputSchema at all.",
        },
        # Must be offered with additionalProperties forced to false.
        {
            "name": "loose_extras",
            "description": "Permits arguments it never declared.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The text."}},
                "additionalProperties": True,
            },
        },
        # Each of the three below carries a keyword bevel's validator does not
        # enforce, so its constraints would silently not apply. Refuse them.
        {
            "name": "ref_tool",
            "description": "Points its schema at a $ref nobody can resolve.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"$ref": "#/definitions/Text"}},
            },
        },
        {
            "name": "any_of_tool",
            "description": "Accepts one of several shapes.",
            "inputSchema": {
                "anyOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "object", "properties": {"b": {"type": "string"}}},
                ]
            },
        },
        {
            "name": "pattern_tool",
            "description": "Constrains a property with a regex.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string", "pattern": "^[a-z]+$"}},
            },
        },
        # Unusable descriptors: refused, and their siblings must survive.
        {
            # No name member at all.
            "description": "Has no name member.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "blank_description",
            "description": "",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "schema_not_object",
            "description": "Sends a schema where a schema should be.",
            "inputSchema": "everything goes",
        },
        {
            "name": "undescribed",
            "description": "Declares a parameter nobody may explain.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        # The pair that collides once both are sanitised.
        _plain_descriptor("my-tool", "Hyphenated twin."),
        _plain_descriptor("my_tool", "Underscored twin."),
    ]


def _tools_payload(options: argparse.Namespace) -> dict[str, object]:
    tools: list[dict[str, object]] = [
        {
            "name": "echo",
            "description": "Return the text you gave me.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
        {
            "name": "boom",
            "description": "Always fail, the way a real tool fails.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "methods",
            "description": "List the JSON-RPC methods this server has received.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "read_note",
            "description": "Read the fixture's one resource.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]
    if getattr(options, "adversarial_tools", False):
        tools.extend(_adversarial_descriptors())
    return {"tools": tools}


def _text_block(text: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}]}


def _echo_text(options: argparse.Namespace, text: str) -> str:
    """What ``echo`` answers: its argument, or the requested environment fact."""
    if getattr(options, "report_env", None) is not None:
        value = os.environ.get(options.report_env)
        shown = "<unset>" if value is None else value
        return f"{options.report_env}={shown}"
    if getattr(options, "report_cwd", False):
        return os.getcwd()
    return str(text)


def _call_tool(name: object, arguments: object, options: argparse.Namespace, state: dict) -> dict:
    if name == "echo":
        text = arguments.get("text", "") if isinstance(arguments, dict) else ""
        return _text_block(_echo_text(options, text))
    if name == "boom":
        result = _text_block("boom")
        result["isError"] = True
        return result
    if name == "methods":
        # Insertion order, deliberately not sorted: the point of this tool is to
        # let a test check the order the client actually sent things in.
        return _text_block(json.dumps(list(state["methods"])))
    if name == "read_note":
        return _text_block(RESOURCE_TEXT)
    if name in CALLABLE_ADVERSARIAL_TOOLS:
        # Echoes what actually arrived, so a test can prove the bridge sent the
        # server's own name on the wire rather than the namespaced one it
        # offered the model, and forwarded the arguments untouched.
        return _text_block(json.dumps({"name": name, "arguments": arguments}))
    return {"isError": True, "content": [{"type": "text", "text": f"unknown tool {name!r}"}]}


def _answer_initialize(request_id: object, options: argparse.Namespace) -> None:
    if options.hang == "initialize":
        return
    result: dict[str, object] = {
        "capabilities": {"tools": {}, "resources": {}},
        "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
    }
    if options.protocol != "omit":
        result["protocolVersion"] = options.protocol
    emit({"jsonrpc": "2.0", "id": request_id, "result": result})


def _answer_tools_list(request_id: object, options: argparse.Namespace) -> None:
    if options.hang == "tools-list":
        return
    frame = json.dumps({"jsonrpc": "2.0", "id": request_id, "result": _tools_payload(options)})
    if options.emit == "split":
        # Half a frame now, the rest later: one os.read is not one frame.
        half = len(frame) // 2
        write_raw(frame[:half])
        time.sleep(0.3)
        write_raw(frame[half:] + "\n")
    elif options.emit == "coalesce":
        noise = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": "coalesced"},
            }
        )
        # One write, two frames: the client's framing has to separate them.
        write_raw(frame + "\n" + noise + "\n")
    else:
        emit({"jsonrpc": "2.0", "id": request_id, "result": _tools_payload(options)})


def _answer_tools_call(
    request_id: object, params: object, options: argparse.Namespace, state: dict
) -> None:
    if options.hang == "call":
        return
    if options.sleep_before_reply:
        time.sleep(options.sleep_before_reply)
    for index in range(options.notify):
        emit(
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": f"noise {index}"},
            }
        )
    if options.garbage_on_call and not state["garbage_sent"]:
        # Once, not on every call: a server that emits one bad line is a
        # server with a bug, and the client has to survive it and keep
        # working. A server that always does this tests nothing past that.
        state["garbage_sent"] = True
        write_raw("this is not json at all\n")
    arguments = params.get("arguments") if isinstance(params, dict) else None
    name = params.get("name") if isinstance(params, dict) else None
    result = _call_tool(name, arguments, options, state)
    if options.is_error and name != "boom":
        result = dict(result)
        result["isError"] = True
    emit({"jsonrpc": "2.0", "id": request_id, "result": result})


def _answer_resources_read(request_id: object, params: object) -> None:
    wanted = params.get("uri") if isinstance(params, dict) else None
    if wanted != RESOURCE_URI:
        emit(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"unknown resource {wanted!r}, expected {RESOURCE_URI!r}",
                },
            }
        )
        return
    emit(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contents": [{"uri": RESOURCE_URI, "mimeType": "text/plain", "text": RESOURCE_TEXT}]
            },
        }
    )


def _answer_resources_list(request_id: object) -> None:
    emit(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": [
                    {
                        "uri": RESOURCE_URI,
                        "name": "notes",
                        "description": "The fixture's one resource.",
                        "mimeType": "text/plain",
                    }
                ]
            },
        }
    )


def _answer(
    method: str, request_id: object, params: object, options: argparse.Namespace, state: dict
) -> None:
    """Reply to one request. Notifications get nothing back, as the spec demands."""
    if method == "initialize":
        _answer_initialize(request_id, options)
    elif method == "tools/list":
        _answer_tools_list(request_id, options)
    elif method == "tools/call":
        _answer_tools_call(request_id, params, options, state)
    elif method == "resources/list":
        _answer_resources_list(request_id)
    elif method == "resources/read":
        _answer_resources_read(request_id, params)
    else:
        emit(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method {method!r} is not implemented"},
            }
        )


def _spawn_grandchild(path: str) -> None:
    """Start the heartbeat grandchild.

    Spawned on the *first request* rather than at startup so the transport has
    certainly finished assigning the server to its Job Object (Windows) or the
    server has certainly become a process-group leader (POSIX) before this
    process exists. A grandchild born before that is outside the tree and
    survives the kill for a reason that has nothing to do with the client.
    """
    subprocess.Popen(
        [sys.executable, "-c", _HEARTBEAT_CODE, path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument("--die-now", action="store_true")
    parser.add_argument("--hang", choices=("startup", "initialize", "tools-list", "call"))
    parser.add_argument("--exit-after", type=int)
    parser.add_argument("--garbage-on-call", action="store_true")
    parser.add_argument("--notify", type=int, default=0)
    parser.add_argument("--stderr-lines", type=int, default=0)
    parser.add_argument("--stderr-bytes", type=int, default=0)
    parser.add_argument("--is-error", action="store_true")
    parser.add_argument(
        "--adversarial-tools",
        action="store_true",
        help="Append the deliberately-broken descriptors to tools/list (m10c).",
    )
    parser.add_argument("--protocol", default=PROTOCOL_VERSION)
    parser.add_argument("--grandchild")
    parser.add_argument("--emit", choices=("whole", "split", "coalesce"), default="whole")
    parser.add_argument("--sleep-before-reply", type=float, default=0.0)
    parser.add_argument(
        "--report-env",
        metavar="NAME",
        help="echo answers with NAME=<value> from this process's environment (m10d).",
    )
    parser.add_argument(
        "--report-cwd",
        action="store_true",
        help="echo answers with this process's working directory (m10d).",
    )
    parser.add_argument("--http", action="store_true", help="serve streamable-HTTP, not stdio.")
    parser.add_argument(
        "--port-file",
        help="with --http: write the bound port here once listening (required).",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="with --http: answer requests with text/event-stream instead of application/json.",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="with --http: never issue an Mcp-Session-Id.",
    )
    parser.add_argument(
        "--bad-session-after",
        type=int,
        metavar="N",
        help="with --http: answer N requests after initialize, then 404 every later one.",
    )
    parser.add_argument(
        "--require-accept",
        action="store_true",
        help="with --http: 400 unless Accept lists both application/json and text/event-stream.",
    )
    parser.add_argument(
        "--require-protocol-header",
        action="store_true",
        help="with --http: 400 unless MCP-Protocol-Version matches the negotiated version.",
    )
    parser.add_argument(
        "--require-header",
        metavar="NAME=VALUE",
        help="with --http: 400 unless that header is present with that value.",
    )
    parser.add_argument(
        "--saw",
        metavar="FILE",
        help="with --http: record what the server actually received, as JSON, after each request.",
    )
    return parser.parse_args(argv)


def _startup_noise(options: argparse.Namespace) -> None:
    """Write whatever stderr the flags ask for, before the first read."""
    if options.stderr_lines:
        for index in range(options.stderr_lines):
            log(f"fixture stderr line {index}")
    if options.stderr_bytes:
        sys.stderr.write("x" * options.stderr_bytes)
        sys.stderr.flush()


def _serve(options: argparse.Namespace) -> int:
    """Read frames until stdin ends. Returns the process exit code."""
    state = {"methods": [], "grandchild_started": False, "garbage_sent": False, "read": 0}

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return 0
        frame = _decode(line)
        if frame is None:
            continue

        method = frame.get("method")
        if isinstance(method, str):
            state["methods"].append(method)
            if not state["grandchild_started"] and options.grandchild:
                state["grandchild_started"] = True
                _spawn_grandchild(options.grandchild)

        if options.exit_after is not None:
            state["read"] += 1
            if state["read"] > options.exit_after:
                log("fixture exiting without answering")
                return 0

        if not isinstance(method, str) or not method:
            log(f"fixture received a frame with no method: {line[:120]!r}")
            continue
        if method not in KNOWN_METHODS:
            log(f"fixture received an unknown method {method!r}")
        if "id" not in frame:
            # A notification. Answering it would be the protocol violation.
            continue
        _answer(method, frame["id"], frame.get("params"), options, state)


def _decode(line: bytes) -> dict | None:
    """One received line as an object, or ``None`` if it is not one."""
    try:
        frame = json.loads(line)
    except json.JSONDecodeError:
        log(f"fixture received a frame that is not JSON: {line[:120]!r}")
        return None
    if not isinstance(frame, dict):
        log(f"fixture received a non-object frame: {line[:120]!r}")
        return None
    return frame


# ---------------------------------------------------------------------------
# Streamable-HTTP mode (m10e).
# ---------------------------------------------------------------------------


class _HttpServer:
    """Everything the handler needs that is not a flag.

    A plain object rather than handler attributes because ``ThreadingHTTPServer``
    answers requests on separate threads and the session, the counters, and the
    observations are shared across them.
    """

    def __init__(self, options: argparse.Namespace) -> None:
        self.options = options
        self.lock = threading.Lock()
        self.session_id: str | None = None
        self.negotiated: str | None = None
        self.initialize_had_session_id = False
        self.session_ids_presented: list[str | None] = []
        self.headers_seen: list[str] = []
        self.deletes = 0
        self.answered_after_initialize = 0
        self.state = {
            "methods": [],
            "grandchild_started": False,
            "garbage_sent": False,
            "read": 0,
        }

    def record(self) -> None:
        """Write what the server has received so far, if ``--saw`` asked for it."""
        if not self.options.saw:
            return
        with self.lock:
            payload = {
                "session_issued": self.session_id,
                "initialize_had_session_id": self.initialize_had_session_id,
                "session_ids_presented": list(self.session_ids_presented),
                "headers_seen": list(self.headers_seen),
                "deletes": self.deletes,
                "methods": list(self.state["methods"]),
            }
        # Written whole, and read whole by the test after a call returns, so the
        # last request's record is the one that lands.
        with open(self.options.saw, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)


class _Server(ThreadingHTTPServer):
    """The threading server, with the fixture's context carried on it.

    ``ThreadingHTTPServer`` builds a handler per request and passes it no
    application state, so the context rides on the server every handler is
    handed. A subclass for one attribute beats a module global that each handler
    has to assert on.
    """

    daemon_threads = True
    ctx: _HttpServer


class _Handler(BaseHTTPRequestHandler):
    """One MCP endpoint speaking streamable-HTTP.

    ``protocol_version`` is 1.1 so httpx's connection reuse works; every response
    sets ``Content-Length`` accordingly, and the ones that end a stream say
    ``Connection: close`` so the client does not wait for a next response that is
    never coming.
    """

    protocol_version = "HTTP/1.1"
    server_version = "bevel-fixture/1.0"

    def do_POST(self) -> None:
        """Handle one JSON-RPC message, which the spec says is one POST."""
        ctx = self.server.ctx  # type: ignore[attr-defined]
        options = ctx.options
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))

        frame = _decode(body)
        if frame is None:
            self._reply(400, b'{"jsonrpc":"2.0","error":{"code":-32700,"message":"bad json"}}')
            return
        method = frame.get("method")
        if not isinstance(method, str) or not method:
            log(f"fixture received an HTTP frame with no method: {body[:120]!r}")
            self._reply(400, b'{"jsonrpc":"2.0","error":{"code":-32600,"message":"no method"}}')
            return

        # Bookkeeping first, and the refusals after it: a request the server
        # rejects is still a request the client sent, and --saw has to show it.
        is_request = "id" in frame
        with ctx.lock:
            ctx.state["methods"].append(method)
            ctx.state["read"] += 1
            if is_request and method != "initialize":
                ctx.session_ids_presented.append(self.headers.get("Mcp-Session-Id"))
            past_limit = (
                ctx.options.exit_after is not None and ctx.state["read"] > ctx.options.exit_after
            )
        if past_limit:
            log("fixture exiting without answering")
            os._exit(0)

        refusal = self._refusal(options, ctx, method, is_request)
        if refusal is not None:
            ctx.record()
            self._reply(*refusal)
            return

        if not is_request:
            # A notification: accepted, and answered with 202 and no body, which is
            # what the spec demands. Answering it would be the protocol violation.
            ctx.record()
            self._reply(202, b"")
            return

        if method == "initialize" and not options.no_session:
            with ctx.lock:
                ctx.session_id = f"fixture-session-{os.getpid()}"

        lines = self._run_handlers(method, frame["id"], frame.get("params"), options, ctx)
        if lines is None:
            # A hang flag produced nothing. Block rather than answer: from the
            # client's side this must be indistinguishable from a server that is
            # alive and thinking, which is the whole point of --hang.
            while True:
                time.sleep(3600)

        with ctx.lock:
            if method != "initialize":
                ctx.answered_after_initialize += 1
            session_for_reply = ctx.session_id if method == "initialize" else None
        ctx.record()

        if options.sse:
            self._reply_sse(lines, session_id=session_for_reply)
        else:
            extra = {"Mcp-Session-Id": session_for_reply} if session_for_reply else {}
            self._reply(200, "".join(lines).encode("utf-8"), "application/json", extra)

    def do_DELETE(self) -> None:
        """End the session. A client that is finished SHOULD send this."""
        ctx = self.server.ctx  # type: ignore[attr-defined]
        with ctx.lock:
            ctx.deletes += 1
        ctx.record()
        self._reply(200, b"")

    def do_GET(self) -> None:
        """No server-initiated stream. 405 is the spec's answer for that."""
        self._reply(405, b"")

    def log_message(self, format: str, *args: object) -> None:
        """Silence the access log; stderr is the fixture's own to write to."""

    # --- internals ----------------------------------------------------------

    def _refusal(  # noqa: PLR0911 - one return per mandatory rule the spec states
        self, options: argparse.Namespace, ctx: _HttpServer, method: str, is_request: bool
    ) -> tuple[int, bytes] | None:
        """The status the spec requires for a request that broke a mandatory rule.

        Enforced rather than echoed: a client that sends ``Accept`` only when it
        remembers, or ``MCP-Protocol-Version`` only after the first call, passes an
        echo and fails here.
        """
        if options.require_accept:
            accept = self.headers.get("Accept", "")
            if "application/json" not in accept or "text/event-stream" not in accept:
                return 400, f'{{"error":"Accept must list both types, got {accept!r}"}}'.encode()
        if options.require_protocol_header and method != "initialize":
            # The spec only mandates this header on requests after the negotiation;
            # on initialize the version is still being chosen, so requiring it there
            # would be the fixture demanding knowledge the client cannot have.
            wanted = options.protocol
            got = self.headers.get("MCP-Protocol-Version")
            if got != wanted:
                return 400, f'{{"error":"MCP-Protocol-Version {got!r}, want {wanted!r}"}}'.encode()
        if options.require_header:
            name, _, value = options.require_header.partition("=")
            got = self.headers.get(name)
            with ctx.lock:
                ctx.headers_seen.append(got if got is not None else "<missing>")
            if got != value:
                return 400, f'{{"error":"header {name!r} is {got!r}, want {value!r}"}}'.encode()

        if method == "initialize":
            with ctx.lock:
                ctx.initialize_had_session_id = self.headers.get("Mcp-Session-Id") is not None
            return None
        if ctx.session_id is None:
            return None
        presented = self.headers.get("Mcp-Session-Id")
        if presented != ctx.session_id:
            return 400, b'{"error":"missing or wrong Mcp-Session-Id"}'
        if is_request and options.bad_session_after is not None:
            with ctx.lock:
                dropped = ctx.answered_after_initialize >= options.bad_session_after
            if dropped:
                return 404, b'{"error":"session terminated"}'
        return None

    def _run_handlers(
        self,
        method: str,
        request_id: object,
        params: object,
        options: argparse.Namespace,
        ctx: _HttpServer,
    ) -> list[str] | None:
        """Run the shared ``_answer_*`` handlers and collect what they emitted.

        ``None`` means the handlers answered nothing, which is how a ``--hang``
        flag reaches HTTP mode: the same code path stdio uses, with no HTTP
        version of the hang to keep in sync.
        """
        _sink.lines = []
        try:
            _answer(method, request_id, params, options, ctx.state)
        finally:
            lines = list(_sink.lines)
            del _sink.lines
        return lines or None

    def _reply(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.wfile.flush()

    def _reply_sse(self, lines: list[str], session_id: str | None = None) -> None:
        """One SSE event per JSON-RPC message, then close the stream.

        The leading ``:`` comment and the ``event: message`` field are both legal
        SSE that real servers send, so they are included deliberately: a client
        that parses SSE by treating every line as data has to fail here rather than
        in front of a server that does it.
        """
        parts = [": fixture keepalive\n\n"]
        for line in lines:
            parts.append(f"event: message\ndata: {line.strip()}\n\n")
        body = "".join(parts).encode("utf-8")
        extra = {"Mcp-Session-Id": session_id} if session_id else {}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in extra.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def _serve_http(options: argparse.Namespace) -> int:
    """Serve the MCP endpoint until killed. Returns only if the bind fails."""
    if not options.port_file:
        log("--http requires --port-file")
        return 2
    ctx = _HttpServer(options)
    try:
        server = _Server(("127.0.0.1", 0), _Handler)
    except OSError as exc:
        log(f"fixture could not bind: {exc}")
        return 2
    server.ctx = ctx
    with open(options.port_file, "w", encoding="utf-8") as handle:
        handle.write(str(server.server_address[1]))
    log(f"fixture listening on port {server.server_address[1]}")
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str]) -> int:
    options = _parse_args(argv)

    if options.hang == "startup":
        # Alive, listening, and never answering: the wire cannot tell this apart
        # from a child that never started, which is why spawn is liveness-based.
        while True:
            time.sleep(3600)

    _startup_noise(options)
    if options.http:
        return _serve_http(options)
    return _serve(options)


if __name__ == "__main__":
    # Checked before argparse so the child dies at interpreter startup, well
    # inside the transport's liveness window; importing argparse first would
    # put its own death late enough to be misread as a handshake failure.
    if "--die-now" in sys.argv:
        os._exit(0)
    try:
        sys.exit(main(sys.argv[1:]))
    except BrokenPipeError:
        # The client closed our stdout and went away. Nothing left to say.
        os._exit(0)

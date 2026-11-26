"""A real stdio MCP server, and the oracle every m10 slice tests against.

This is not a test module and must never import pytest. It is a plain script
runnable as ``[sys.executable, "tests/fixtures/mcp_server.py", ...]``, which is
how :mod:`bevel.mcp.stdio` spawns it in ``tests/unit/test_mcp_stdio.py`` and how
every later m10 slice will spawn it too. m10a could prove pairing in memory;
nothing after m10b can be believed unless it is proven against a process that
actually holds a pipe.

Everything a test needs to misbehave is a flag, so the failure mode lives in
the server rather than in a monkeypatch of the client. A transport bug and a
test-double bug look identical; only a real child can tell them apart.

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
import time

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


def emit(payload: object) -> None:
    """Write one JSON-RPC frame: one line, delimiter included, flushed."""
    write_raw(json.dumps(payload) + "\n")


def write_raw(text: str) -> None:
    """Write bytes to stdout exactly as given, then flush.

    Raw rather than ``print`` because two of the behaviours under test are
    "half a frame" and "two frames in one write", and a line-buffered text
    stream would silently rewrite both.
    """
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


def main(argv: list[str]) -> int:
    options = _parse_args(argv)

    if options.hang == "startup":
        # Alive, listening, and never answering: the wire cannot tell this apart
        # from a child that never started, which is why spawn is liveness-based.
        while True:
            time.sleep(3600)

    _startup_noise(options)
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

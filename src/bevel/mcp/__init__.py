"""Model Context Protocol client.

What exists today is the wire and one way to carry it.
:mod:`bevel.mcp.jsonrpc` is the JSON-RPC 2.0 core (m10a): request, notification
and response envelopes, id allocation, the spec's error codes, and
request/response pairing over an abstract
:class:`~bevel.mcp.jsonrpc.DuplexStream`. It performs no I/O at all, which is
the point: the whole of it is proven by feeding bytes to an in-memory stream, so
the framing and pairing decisions every later slice depends on were checked
before anything could hang on a pipe.

:mod:`bevel.mcp.stdio` is the m10b transport. It spawns a server by argv with
``create_subprocess_exec`` and an explicit argv list (never a shell), speaks
newline-delimited JSON-RPC to its stdin and stdout, drains both pipes from
background tasks so a chatty server can never block on a full pipe, keeps a
bounded tail of stderr to quote in every error, negotiates the protocol version
during ``initialize`` → ``notifications/initialized`` → ``tools/list``, applies
separate spawn / handshake / call timeouts whose messages name the stage and
what it waited for, and tears the whole process tree down through
:class:`~bevel.core._process.TreeKiller` on every exit path, including the ones
an exception or a cancellation takes.

It is tested against ``tests/fixtures/mcp_server.py`` — a real stdio MCP server
in this repo, spawned by argv, that answers ``initialize``, ``tools/list``,
``tools/call``, ``resources/list`` and ``resources/read`` and can be told to
hang, exit early, emit garbage, write to stderr, or return ``isError``. That
fixture is the oracle every later m10 slice reuses; see
``tests/unit/test_mcp_stdio.py`` for how it is driven.

What does not exist yet, and the slice that delivers each:

``bridge.py``
    m10c -- MCP tool descriptors into ``ToolSpec`` at the bevel tool seam,
    namespaced and sanitised names, ``isError`` into a failed ``ToolResult``.
    Bridged tools are injected, never registered in ``core/tools/registry.py``.
    What it gets from m10b is :meth:`~bevel.mcp.stdio.StdioTransport.start`
    (a negotiated, ready-to-use transport) and the raw ``tools/list`` dicts it
    returns; the translation is entirely m10c's.
config and ``bevel mcp``
    m10d -- ``[mcp.servers.<name>]`` and ``bevel mcp list``. This is the slice
    that removes ``"mcp"`` from ``_PENDING`` in ``cli.py``; until then
    ``bevel mcp`` exits non-zero and says so, which is what it still does.
``http.py``
    m10e -- streamable-HTTP transport, session ids and SSE. It implements the
    same two-method :class:`~bevel.mcp.jsonrpc.DuplexStream`, so the channel and
    every fixture-driven expectation above carry over.
``resources.py``
    m10f -- ``resources/list`` and ``resources/read``, plus the two
    model-facing tools that actually put a fetched resource in front of the
    model.

The plan stays inside the three-runtime-dependency rule: JSON-RPC is
hand-rolled rather than the official SDK, which pulls in ``pydantic-core`` (a
compiled dependency). Only the subset bevel needs is implemented:
``initialize``, ``tools/list``, ``tools/call``, ``resources/list``,
``resources/read``.
"""

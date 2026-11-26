"""Model Context Protocol client.

What exists today is :mod:`bevel.mcp.jsonrpc` -- the JSON-RPC 2.0 core (m10a):
request, notification and response envelopes, id allocation, the spec's error
codes, and request/response pairing over an abstract
:class:`~bevel.mcp.jsonrpc.DuplexStream`. It performs no I/O at all, which is
the point: the whole of it is proven by feeding bytes to an in-memory stream,
so the framing and pairing decisions every later slice depends on were checked
before anything could hang on a pipe.

What does not exist yet, and the slice that delivers each:

``stdio.py``
    m10b -- spawn by argv, newline-delimited framing, drained stderr, the
    ``initialize`` handshake and version negotiation, per-stage timeouts,
    kill-tree teardown. Also ships ``tests/fixtures/mcp_server.py``, the real
    server every later slice tests against.
``bridge.py``
    m10c -- MCP tool descriptors into ``ToolSpec`` at the bevel tool seam,
    namespaced and sanitised names, ``isError`` into a failed ``ToolResult``.
    Bridged tools are injected, never registered in ``core/tools/registry.py``.
config and ``bevel mcp``
    m10d -- ``[mcp.servers.<name>]`` and ``bevel mcp list``. This is the slice
    that removes ``"mcp"`` from ``_PENDING`` in ``cli.py``; until then
    ``bevel mcp`` exits non-zero and says so, which is what it still does.
``http.py``
    m10e -- streamable-HTTP transport, session ids and SSE.
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

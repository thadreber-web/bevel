"""Model Context Protocol client.

What exists today is the wire, one way to carry it, and the bridge that turns a
server's advertised tools into tools the model is offered.
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
``tests/unit/test_mcp_stdio.py`` for how it is driven. With
``--adversarial-tools`` it also advertises descriptors that a well-behaved server
would not -- a name with spaces and punctuation, a missing ``inputSchema``, a
schema that permits extras, ``$ref``/``anyOf``/``pattern`` schemas, a nameless
descriptor, a blank description, a non-object schema, an undocumented property and
a pair of names that sanitise to the same tool -- which is what
``tests/unit/test_mcp_bridge.py`` drives the bridge against.

:mod:`bevel.mcp.bridge` is m10c, the tool bridge. :func:`~bevel.mcp.bridge.bridge_tools`
takes a transport someone else already started, issues its own ``tools/list``, and
returns objects satisfying :class:`~bevel.core.tools.base.Tool` -- so this is the
one module in the package allowed to cross from the wire into ``core/tools/``.
Offered names are ``server__tool``, sanitised to the ``^[a-z][a-z0-9_]*$`` the
registry pins, because one unsanitised remote name is a schema the endpoint rejects
at *offer* time and takes the whole list down with it. ``additionalProperties`` is
forced to ``false``; a descriptor whose ``inputSchema`` uses keywords bevel's
validator ignores is refused rather than offered with a constraint that would not
be applied; a broken descriptor costs that one tool and not its siblings, and the
reason travels to the caller in :class:`~bevel.mcp.bridge.BridgedServer.refusals`.
Every bridged tool is ``EXEC_UNSAFE`` and serial with no per-tool override, and
``isError`` comes back as a failed ``ToolResult`` so the model reads the failure
instead of the iteration aborting. Bridged tools are injected via
``Agent(tools=...)`` and are never registered in ``core/tools/registry.py``, which
stays the source of truth for built-ins only.

What does not exist yet, and the slice that delivers each:

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

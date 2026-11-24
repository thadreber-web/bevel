"""Model Context Protocol client -- **not implemented. This package is empty until m10.**

``bevel mcp`` exits non-zero and says so. The plan, which has to stay inside the
three-runtime-dependency rule: hand-rolled JSON-RPC over stdio and streamable-HTTP
rather than the official SDK, which pulls in ``pydantic-core`` (a compiled
dependency). Implement the subset bevel needs: ``initialize``, ``tools/list``,
``tools/call``, ``resources/list``, ``resources/read``.
"""

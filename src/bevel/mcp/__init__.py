"""Model Context Protocol client.

Hand-rolled JSON-RPC over stdio and streamable-HTTP rather than the official
SDK, which pulls in ``pydantic-core`` (a compiled dependency). We implement the
subset bevel needs: ``initialize``, ``tools/list``, ``tools/call``,
``resources/list``, ``resources/read``.
"""

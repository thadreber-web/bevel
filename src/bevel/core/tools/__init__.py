"""Tools the agent can call.

Each tool declares a JSON Schema, a risk tier, and an async ``execute``. The
policy engine decides before anything runs. Containment today is the T0 path
jail in :class:`~bevel.core.tools.base.ToolContext`; kernel enforcement is m12.
"""

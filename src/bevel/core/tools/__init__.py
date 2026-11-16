"""Tools the agent can call.

Each tool declares a JSON Schema, a risk tier, and an async ``execute``. The
policy engine decides before anything runs; the sandbox constrains how it runs.
"""

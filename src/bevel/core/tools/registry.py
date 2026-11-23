"""The single source of truth for the tools the model is offered.

``core/tools/types.py`` points here: a tool that exists but is not listed is
invisible to the model, and a tool listed twice is a schema the endpoint will
reject. So the list is built once, in one place, in a deliberate order --
read-only tools first, then the ones that touch the workspace, then the one that
runs programs -- and nothing else gets to assemble its own tool list.

Adding a tool means importing it above and appending an instance below. The
tests in ``tests/unit/test_tools_registry.py`` reject a schema the validator
does not fully understand, an undocumented property, or a tier that m03 has no
producer for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from bevel.core.tools.base import Tool
from bevel.core.tools.edit import EditTool
from bevel.core.tools.glob import GlobTool
from bevel.core.tools.grep import GrepTool
from bevel.core.tools.memory import RecallTool, RememberTool
from bevel.core.tools.read import ReadTool
from bevel.core.tools.shell import ShellTool
from bevel.core.tools.types import ToolSpec
from bevel.core.tools.write import WriteTool

#: The offered tools, in the order they are presented to the model.
_TOOLS: Final[tuple[Tool, ...]] = (
    ReadTool(),
    GrepTool(),
    GlobTool(),
    RecallTool(),
    WriteTool(),
    EditTool(),
    RememberTool(),
    ShellTool(),
)

_BY_NAME: Final[dict[str, Tool]] = {tool.spec.name: tool for tool in _TOOLS}


def tools() -> Sequence[Tool]:
    """Every tool the agent may offer, in presentation order."""
    return _TOOLS


def get(name: str) -> Tool | None:
    """Look a tool up by its wire name, or ``None`` if it is not offered."""
    return _BY_NAME.get(name)


def specs() -> list[ToolSpec]:
    """The wire-facing declarations, in the same order as :func:`tools`."""
    return [tool.spec for tool in _TOOLS]


__all__ = ["get", "specs", "tools"]

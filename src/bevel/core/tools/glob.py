"""The ``glob`` tool: list files by pattern.

Patterns are matched against workspace-relative POSIX paths, because that is
the form ``grep`` prints and the form a model feeds back into ``read`` and
``edit``. The matching rules are the ones :mod:`bevel.core.tools._fs` defines
for ``grep --glob`` too, so a pattern that selects a file in one tool selects
it in the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from bevel.core.tools import _fs
from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

DEFAULT_MAX_RESULTS: Final[int] = 500
MAX_RESULTS_CAP: Final[int] = 5000


class GlobTool:
    """Find files in the workspace by glob pattern."""

    spec: Final[ToolSpec] = ToolSpec(
        name="glob",
        description=(
            "Find files in the workspace by glob pattern and return their paths, "
            "sorted, relative to the workspace with forward slashes. A pattern "
            "without a '/' matches file names anywhere ('*.py'); one with a '/' "
            "matches the end of the path ('util/*.py'). '*' and '?' stop at "
            "directory boundaries, '**' crosses them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '*.py', 'src/**/*.ts', 'test_*.py'.",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (default: the whole workspace).",
                },
                "max_results": {
                    "type": "integer",
                    "description": (f"Maximum paths to return (default {DEFAULT_MAX_RESULTS})."),
                    "minimum": 1,
                    "maximum": MAX_RESULTS_CAP,
                    "default": DEFAULT_MAX_RESULTS,
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.READ
    is_parallel_safe: Final[bool] = True

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = arguments["pattern"]
        if not pattern.strip():
            raise ToolError("'pattern' must not be empty")

        root = ctx.require_within_workspace(arguments.get("path", "."))
        if not root.is_dir():
            raise ToolError(f"no such directory: {ctx.display(root)}")

        max_results = min(int(arguments.get("max_results", DEFAULT_MAX_RESULTS)), MAX_RESULTS_CAP)

        hits: list[str] = []
        truncated = False
        for path in _fs.iter_files(root):
            rel = _fs.relative_to_root(path, ctx.workspace)
            if not _fs.glob_matches(pattern, rel):
                continue
            if len(hits) >= max_results:
                truncated = True
                break
            hits.append(rel)

        if not hits:
            where = ctx.display(root)
            return ToolResult(ok=True, content=f"glob: no match for {pattern!r} under {where}.")

        hits.sort()
        if truncated:
            hits.append(
                f"[stopped at {max_results} paths; raise max_results or narrow the pattern]"
            )
        return ToolResult(ok=True, content="\n".join(hits))


__all__ = ["GlobTool"]

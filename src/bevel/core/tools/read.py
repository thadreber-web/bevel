"""The ``read`` tool: numbered lines, streamed, with honest refusals.

Line numbers are the whole point. A model cites ``src/app.py:42`` back at
``edit``, so the numbering has to be the file's real numbering regardless of
where the read started, and a read that silently returns nothing because the
offset ran past the end is worse than one that says so.

The file is streamed rather than slurped: the workspace is somebody's real
repository, and a multi-megabyte log has to be readable by offset without the
agent allocating the whole file to do it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from bevel.core.tools._fs import is_binary
from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

#: Most lines a single call may return. A model cannot use a 10k-line dump, and
#: the round trip would cost more than the answer is worth.
MAX_LINES: Final[int] = 2000
DEFAULT_LINES: Final[int] = 200
#: Characters kept from a single line. Minified bundles and base64 blobs are the
#: common case; the tail of those lines is never what was being looked for.
MAX_LINE_CHARS: Final[int] = 2000
#: Bytes kept from a single call's output, whatever the line settings produced.
MAX_OUTPUT_BYTES: Final[int] = 128_000
#: Bytes sniffed for a NUL before deciding the file is binary.
_SNIFF_BYTES: Final[int] = 8192


class ReadTool:
    """Read a text file from the workspace."""

    spec: Final[ToolSpec] = ToolSpec(
        name="read",
        description=(
            "Read a text file from the workspace and return it with line numbers "
            "as `N|text`. Use offset and limit to page through large files; the "
            "output names the offset to continue from. Refuses binary files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to read, relative to the workspace.",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start at.",
                    "minimum": 1,
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum lines to return (default {DEFAULT_LINES}).",
                    "minimum": 1,
                    "maximum": MAX_LINES,
                    "default": DEFAULT_LINES,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.READ
    is_parallel_safe: Final[bool] = True

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.require_within_workspace(arguments["path"])
        offset = int(arguments.get("offset", 1))
        limit = min(int(arguments.get("limit", DEFAULT_LINES)), MAX_LINES)

        if not path.exists():
            raise ToolError(f"no such file: {ctx.display(path)}")
        if path.is_dir():
            raise ToolError(
                f"{ctx.display(path)} is a directory. Use glob to list it, or read a file."
            )

        with path.open("rb") as handle:
            if is_binary(handle.read(_SNIFF_BYTES)):
                raise ToolError(
                    f"{ctx.display(path)} is a binary file, so read will not dump it. "
                    "Use shell with a purpose-built command to inspect it."
                )

        selected: list[str] = []
        total_lines = 0
        more_remain = False
        # utf-8-sig strips a BOM when present and decodes plain UTF-8 otherwise.
        # errors="replace" because one bad byte in a 10k-line file should not
        # cost the model the whole file.
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                total_lines = number
                if number < offset:
                    continue
                if len(selected) >= limit:
                    more_remain = True
                    break
                selected.append(line.rstrip("\r\n"))

        if total_lines == 0:
            return ToolResult(ok=True, content=f"{ctx.display(path)} is an empty file.")
        if not selected:
            plural = "" if total_lines == 1 else "s"
            return ToolResult.failure(
                f"read: {ctx.display(path)} has {total_lines} line{plural}; "
                f"offset {offset} is past the end."
            )

        lines = [_format_line(offset + index, text) for index, text in enumerate(selected)]
        if more_remain:
            lines.append(
                f"[{len(selected)} of {total_lines}+ lines shown; more remain - "
                f"continue with read offset={offset + limit}]"
            )
        return _limit_bytes("\n".join(lines), MAX_OUTPUT_BYTES)


def _format_line(number: int, text: str) -> str:
    if len(text) > MAX_LINE_CHARS:
        return f"{number}|{text[:MAX_LINE_CHARS]} ... [line truncated at {MAX_LINE_CHARS} chars]"
    return f"{number}|{text}"


def _limit_bytes(content: str, cap: int) -> ToolResult:
    if len(content.encode("utf-8")) <= cap:
        return ToolResult(ok=True, content=content)
    kept = content.encode("utf-8")[:cap].decode("utf-8", errors="replace")
    return ToolResult(
        ok=True,
        content=f"{kept}\n[output truncated at {cap} bytes; narrow the offset or limit]",
    )


__all__ = ["ReadTool"]

"""The ``write`` tool: create or replace a whole file.

Whole-file writes are the blunt instrument; ``edit`` is the precise one. The
model is offered both because some changes (a new file, a generated file) have
no sensible diff to start from, and because a tool that can only patch cannot
create.

Two rules make this safe enough to offer at ``WRITE_WORKSPACE``: an existing
file is never clobbered without ``overwrite``, and the write is atomic, so a
crash or a disk-full mid-write cannot leave a half-written source file where a
good one used to be.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec


class WriteTool:
    """Write a text file in the workspace."""

    spec: Final[ToolSpec] = ToolSpec(
        name="write",
        description=(
            "Create a new file, or replace an existing one when overwrite is true. "
            "Content is written exactly as given, including line endings. Parent "
            "directories are created. Returns a unified diff of the change. Prefer "
            "edit for changing part of an existing file."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to write, relative to the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete new content of the file.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Replace the file if it already exists.",
                    "default": False,
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.WRITE_WORKSPACE
    is_parallel_safe: Final[bool] = False

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.require_within_workspace(arguments["path"])
        content: str = arguments["content"]
        overwrite = bool(arguments.get("overwrite", False))

        if path.is_dir():
            raise ToolError(f"{ctx.display(path)} is a directory, not a file")
        if path.exists() and not overwrite:
            raise ToolError(
                f"{ctx.display(path)} already exists. Pass overwrite=true to replace it, "
                "or use edit to change part of it."
            )

        previous = _read_text(path) if path.exists() else None
        if previous == content:
            return ToolResult(ok=True, content=f"{ctx.display(path)}: no change.")

        _atomic_write(path, content)

        rel = ctx.display(path)
        if previous is None:
            added = len(content.splitlines())
            return ToolResult(
                ok=True, content=f"created {rel} ({added} line{'s' if added != 1 else ''})."
            )
        return ToolResult(ok=True, content=_diff(rel, previous, content) or f"{rel}: no change.")


def _read_text(path: Path) -> str:
    # utf-8-sig so a BOM does not become invisible content in the diff, and
    # errors="replace" so an odd byte in the old file cannot block the write.
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return handle.read()


def _atomic_write(path: Path, content: str) -> None:
    """Write via a sibling temp file and ``os.replace``.

    ``os.replace`` is atomic within a directory on both platforms, so readers
    see either the old file or the new one. newline="" keeps the content's own
    line endings -- translating them on Windows would rewrite every line of a
    file the model only meant to touch in one place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        # Only removes the temp file when the replace did not happen; after a
        # successful replace the name is gone anyway.
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _diff(rel: str, before: str, after: str) -> str:
    """A unified diff, or empty text when the two are identical."""
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=rel,
        tofile=rel,
        lineterm="",
    )
    body = "\n".join(lines)
    if not body:
        return ""
    changed = sum(1 for line in lines if line[:1] in "+-" and not line.startswith(("---", "+++")))
    return f"{rel}: {changed} line{'s' if changed != 1 else ''} changed\n{body}"


__all__ = ["WriteTool"]

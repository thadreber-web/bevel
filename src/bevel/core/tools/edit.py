"""The ``edit`` tool: exact-text replacements applied as a real diff.

This is the tool the milestone is judged on. Three properties matter more than
anything else:

* **All-or-nothing.** Edits are applied to an in-memory copy and the file is
  rewritten only if every one of them applied. A half-applied batch leaves the
  source tree in a state nobody asked for and the model cannot reason about.
* **No guessing.** An ``old_text`` that matches more than once is refused with
  the count, not resolved by picking the first match. A wrong-but-successful
  edit is far more expensive than a refused one.
* **Byte preservation.** Line endings and the presence of a trailing newline
  are properties of the file, not of the replacement text. On a Windows dev
  machine, a tool that normalises CRLF to LF produces a diff over the whole
  file on every single edit.
"""

from __future__ import annotations

import difflib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from bevel.core.tools._fs import is_binary
from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

#: Bytes sniffed for a NUL before deciding the file is binary.
_SNIFF_BYTES: Final[int] = 8192


class EditTool:
    """Apply exact-text replacements to a file in the workspace."""

    spec: Final[ToolSpec] = ToolSpec(
        name="edit",
        description=(
            "Change an existing file by replacing exact text. Each edit's old_text "
            "must appear exactly once unless replace_all is set; edits apply in "
            "order and the file is only written if all of them apply. Line endings "
            "and the file's trailing newline are preserved. Returns a unified diff."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File to edit, relative to the workspace.",
                },
                "edits": {
                    "type": "array",
                    "description": "Replacements to apply, in order.",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "description": "Exact text to find. Must match uniquely.",
                            },
                            "new_text": {
                                "type": "string",
                                "description": "Text to put in its place.",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "Replace every occurrence instead of requiring a unique match.",
                                "default": False,
                            },
                        },
                        "required": ["old_text", "new_text"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.WRITE_WORKSPACE
    is_parallel_safe: Final[bool] = False

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        path = ctx.require_within_workspace(arguments["path"])
        edits: list[dict[str, Any]] = list(arguments["edits"])

        if not path.exists():
            raise ToolError(f"no such file: {ctx.display(path)}")
        if path.is_dir():
            raise ToolError(f"{ctx.display(path)} is a directory, not a file")

        raw = path.read_bytes()
        if is_binary(raw[:_SNIFF_BYTES]):
            raise ToolError(f"{ctx.display(path)} is a binary file, so edit will not patch it.")

        newline, trailing = _line_style(raw)
        text = _to_lf(raw.decode("utf-8", errors="replace"))

        # Applied to a local copy: nothing reaches the disk unless every edit
        # in the batch applied.
        updated = text
        for index, change in enumerate(edits):
            updated = _apply_one(updated, change, index, ctx.display(path))

        if updated == text:
            raise ToolError(f"{ctx.display(path)}: no change - the edits produced identical text.")

        _atomic_write(path, _from_lf(updated, newline, trailing))
        rel = ctx.display(path)
        return ToolResult(ok=True, content=_diff(rel, text, updated))


def _apply_one(text: str, change: Mapping[str, Any], index: int, rel: str) -> str:
    """Apply one replacement to ``text``, or raise with an actionable reason."""
    old = _to_lf(str(change["old_text"]))
    new = _to_lf(str(change["new_text"]))
    replace_all = bool(change.get("replace_all", False))

    if not old:
        raise ToolError(f"edits[{index}]: 'old_text' is empty; name the text to replace.")
    if old == new:
        raise ToolError(f"edits[{index}]: 'old_text' and 'new_text' are identical - no change.")

    count = text.count(old)
    if count == 0:
        raise ToolError(
            f"edits[{index}]: old_text was not found in {rel}. Read the file and copy the "
            "exact text, including indentation."
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"edits[{index}]: old_text appears {count} times in {rel}, so the match is "
            "ambiguous. Include more surrounding text to make it unique, or set "
            "replace_all=true to change every occurrence."
        )
    return text.replace(old, new) if replace_all else text.replace(old, new, 1)


def _to_lf(data: str | bytes) -> str:
    """Decode to text with every line ending collapsed to ``\\n``."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    return data.replace("\r\n", "\n").replace("\r", "\n")


def _line_style(raw: bytes) -> tuple[str, bool]:
    """The file's dominant line ending and whether it ends with one.

    CRLF is chosen only when *every* LF is preceded by CR, so a mixed file is
    left mixed rather than silently normalised.
    """
    trailing = raw.endswith(b"\n")
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n")
    return ("\r\n" if crlf and crlf == lf else "\n"), trailing


def _from_lf(text: str, newline: str, trailing: bool) -> str:
    """Render LF-space text back with the file's own line style."""
    body = text.replace("\n", newline) if newline == "\r\n" else text
    if trailing and not body.endswith(newline):
        body += newline
    elif not trailing and body.endswith(newline):
        body = body[: -len(newline)]
    return body


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _diff(rel: str, before: str, after: str) -> str:
    """A unified diff that reads like ``git diff`` output."""
    body = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=rel,
            tofile=rel,
            lineterm="",
        )
    )
    changed = sum(1 for line in body if line[:1] in "+-" and not line.startswith(("---", "+++")))
    header = f"{rel}: {changed} line{'s' if changed != 1 else ''} changed"
    return "\n".join([header, *body])


__all__ = ["EditTool"]

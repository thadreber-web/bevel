"""The ``grep`` tool: a regex search over the workspace, in pure Python.

A bundled ripgrep binary would be a compiled dependency, which the
three-dependency rule forbids, so this walks the tree itself. That is fine for
the sizes a coding agent actually searches, and it means the output format is
chosen for the model rather than inherited from a subprocess.

Output is ``path:line:text`` for a hit and ``path-line-text`` for a context
line -- the ripgrep convention, because those strings are what a model feeds
straight back into ``read`` and ``edit``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

from bevel.core.tools import _fs
from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

DEFAULT_MAX_RESULTS: Final[int] = 200
MAX_RESULTS_CAP: Final[int] = 2000
#: Largest file to read. Anything bigger is a build artefact or a log, and a
#: regex over it is not what the model meant.
MAX_FILE_BYTES: Final[int] = 4_000_000
#: Bytes sniffed for a NUL before deciding the file is binary.
_SNIFF_BYTES: Final[int] = 8192
#: Characters kept from a single matched line.
MAX_LINE_CHARS: Final[int] = 500


class GrepTool:
    """Search file contents with a regular expression."""

    spec: Final[ToolSpec] = ToolSpec(
        name="grep",
        description=(
            "Search file contents in the workspace with a Python regular expression. "
            "Returns matching lines as `path:line:text`; set context for surrounding "
            "lines. Skips binary files and directories like .git and node_modules."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Subdirectory or file to search (default: the whole workspace).",
                },
                "glob": {
                    "type": "string",
                    "description": (
                        "Only search files whose path matches this glob, e.g. '*.py' or 'util/*.py'."
                    ),
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "Match case-insensitively.",
                    "default": False,
                },
                "context": {
                    "type": "integer",
                    "description": "Lines of context to include before and after each hit.",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 0,
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Maximum matching lines to return (default {DEFAULT_MAX_RESULTS})."
                    ),
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
        flags = re.IGNORECASE if arguments.get("ignore_case") else 0
        try:
            regex = re.compile(arguments["pattern"], flags)
        except re.error as exc:
            raise ToolError(f"pattern is not a valid regular expression: {exc}") from exc

        root = ctx.require_within_workspace(arguments.get("path", "."))
        if not root.exists():
            raise ToolError(f"no such file or directory: {ctx.display(root)}")

        glob_pattern: str | None = arguments.get("glob")
        max_results = min(int(arguments.get("max_results", DEFAULT_MAX_RESULTS)), MAX_RESULTS_CAP)
        context = int(arguments.get("context", 0))

        out: list[str] = []
        truncated = False
        searched = 0
        skipped_binary = 0
        skipped_large = 0

        for path in _search_files(root, glob_pattern):
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped_large += 1
                    continue
                with path.open("rb") as handle:
                    if _fs.is_binary(handle.read(_SNIFF_BYTES)):
                        skipped_binary += 1
                        continue
                produced = _search_file(path, regex, ctx.relative(path), context)
            except OSError:
                # A file that vanishes or denies us mid-scan is a skipped file,
                # not a failed search: the answer to the question is still valid.
                continue
            searched += 1
            if len(out) + len(produced) > max_results:
                out.extend(produced[: max(max_results - len(out), 0)])
                truncated = True
                break
            out.extend(produced)

        if not out:
            return ToolResult(ok=True, content=_no_match_note(searched, skipped_binary))
        if truncated:
            out.append(
                f"[stopped at {max_results} matching lines; raise max_results or narrow the search]"
            )
        note = _skip_note(skipped_binary, skipped_large)
        if note:
            out.append(note)
        return ToolResult(ok=True, content="\n".join(out))


def _search_files(root: Path, glob_pattern: str | None) -> Iterator[Path]:
    """Files to search, honouring ``path=`` pointing at a single file."""
    if root.is_file():
        if glob_pattern is None or _fs.glob_matches(glob_pattern, root.name):
            yield root
        return
    for path in _fs.iter_files(root):
        rel = _fs.relative_to_root(path, root)
        if glob_pattern is None or _fs.glob_matches(glob_pattern, rel):
            yield path


def _search_file(path: Path, regex: re.Pattern[str], rel: str, context: int) -> list[str]:
    """Every ``path:line:text`` and context line for one file."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = [line.rstrip("\r\n") for line in handle]

    out: list[str] = []
    rendered: set[int] = set()
    for index, line in enumerate(lines):
        if not regex.search(line):
            continue
        # Overlapping context blocks share lines; render each line once, and
        # only ever as a hit if it is itself a hit.
        for number in range(max(0, index - context), min(len(lines) - 1, index + context) + 1):
            if number in rendered:
                continue
            rendered.add(number)
            separator = ":" if number == index or regex.search(lines[number]) else "-"
            out.append(f"{rel}{separator}{number + 1}{separator}{_clip(lines[number])}")
    return out


def _clip(text: str) -> str:
    if len(text) > MAX_LINE_CHARS:
        return f"{text[:MAX_LINE_CHARS]} ... [line truncated]"
    return text


def _no_match_note(searched: int, skipped_binary: int) -> str:
    if searched == 0:
        return "grep: no match. No text files were available to search."
    files = "" if searched == 1 else "s"
    binary = "" if skipped_binary == 0 else f", {skipped_binary} binary file(s) skipped"
    return f"grep: no match. Searched {searched} file{files}{binary}."


def _skip_note(skipped_binary: int, skipped_large: int) -> str | None:
    parts: list[str] = []
    if skipped_binary:
        parts.append(f"{skipped_binary} binary file(s) skipped")
    if skipped_large:
        parts.append(f"{skipped_large} file(s) over {MAX_FILE_BYTES} bytes skipped")
    return f"[{', '.join(parts)}]" if parts else None


__all__ = ["GrepTool"]

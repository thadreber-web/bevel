"""Parse and render unified diffs.

The tools in :mod:`bevel.core.tools` produce diff text with
:func:`difflib.unified_diff` and a one-line summary header. This module turns
that text into structure the TUI can render, and renders it for a terminal.

Everything here is a pure function over strings so it can be unit-tested
without a Textual Pilot. Rendering never raises: a diff this module cannot
parse comes back with :attr:`ParsedDiff.error` set and renders as a visible
failure. An empty box that looks like "no changes" is the worst thing a diff
viewer can do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.text import Text

__all__ = [
    "DiffLine",
    "DiffLineKind",
    "ParsedDiff",
    "parse_diff",
    "render_diff",
]


DiffLineKind = str
"""One of ``add``/``del``/``ctx``/``hunk``/``header``/``raw``."""

ADD = "add"
DELETE = "del"
CONTEXT = "ctx"
HUNK = "hunk"
HEADER = "header"
RAW = "raw"

_HUNK_PREFIX = "@@ -"
_OLD_FILE_PREFIX = "--- "
_NEW_FILE_PREFIX = "+++ "

# marker is None for lines that carry no gutter and are appended verbatim.
_STYLE_BY_KIND: dict[DiffLineKind, tuple[str, str | None]] = {
    HUNK: ("bold cyan", None),
    HEADER: ("dim", None),
    RAW: ("dim italic", None),
    ADD: ("green", "+"),
    DELETE: ("red", "-"),
    CONTEXT: ("", " "),
}


@dataclass(frozen=True)
class DiffLine:
    """One rendered line of a diff, with its old and new line numbers."""

    kind: DiffLineKind
    text: str
    old_line: int | None = None
    new_line: int | None = None


@dataclass
class ParsedDiff:
    """A unified diff broken into lines, with counts and a failure note."""

    path: str | None = None
    summary: str = ""
    lines: list[DiffLine] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the text parsed into at least one diff line."""
        return self.error is None and bool(self.lines)

    @property
    def is_empty(self) -> bool:
        """True when the diff genuinely has no hunks."""
        return self.error is None and not self.lines


@dataclass
class _Cursor:
    """Line counters advanced as :func:`parse_diff` walks the body."""

    old_no: int | None = None
    new_no: int | None = None
    added: int = 0
    removed: int = 0
    saw_hunk: bool = False


def parse_diff(text: str) -> ParsedDiff:
    """Parse unified-diff text.

    Accepts the output of :func:`difflib.unified_diff` with or without the
    ``path: N line(s) changed`` header the write and edit tools prepend. Never
    raises; a malformed or empty input yields a :class:`ParsedDiff` with
    :attr:`~ParsedDiff.error` set.
    """
    if not text or not text.strip():
        return ParsedDiff(error="no diff text was provided")

    raw_lines = text.splitlines()
    summary, path, index = _read_preamble(raw_lines)

    lines: list[DiffLine] = []
    cursor = _Cursor()

    for offset in range(index, len(raw_lines)):
        line = raw_lines[offset]
        if line.startswith(_HUNK_PREFIX):
            old_no, new_no = _parse_hunk_header(line)
            if old_no is None and new_no is None:
                return ParsedDiff(error=f"malformed hunk header: {line!r}")
            cursor.old_no, cursor.new_no, cursor.saw_hunk = old_no, new_no, True
            lines.append(DiffLine(HUNK, line))
        elif line.startswith((_OLD_FILE_PREFIX, _NEW_FILE_PREFIX)):
            path = path or (line[4:].strip() or None)
            lines.append(DiffLine(HEADER, line))
        elif not cursor.saw_hunk:
            # Junk between the summary and the first hunk; keep it visible
            # rather than dropping it, but do not count it as a change.
            if line.strip():
                lines.append(DiffLine(RAW, line))
        else:
            lines.append(_body_line(line, cursor))

    if not cursor.saw_hunk:
        return ParsedDiff(
            path=path,
            summary=summary,
            lines=lines,
            error="no hunk headers found; this is not a unified diff",
        )
    return ParsedDiff(
        path=path,
        summary=summary,
        lines=lines,
        added=cursor.added,
        removed=cursor.removed,
    )


def _read_preamble(raw_lines: list[str]) -> tuple[str, str | None, int]:
    """Return the summary line, the path it names, and where the body starts."""
    if raw_lines[0].startswith((_HUNK_PREFIX, _OLD_FILE_PREFIX)):
        return "", None, 0
    summary = raw_lines[0].strip()
    head, _, _rest = summary.partition(":")
    return summary, (head if _looks_like_path(head) else None), 1


def _body_line(line: str, cursor: _Cursor) -> DiffLine:
    """Build the diff line for a body line and advance the cursor."""
    if line.startswith("+"):
        cursor.added += 1
        return DiffLine(ADD, line[1:], None, _bump(cursor.new_no))
    if line.startswith("-"):
        cursor.removed += 1
        return DiffLine(DELETE, line[1:], _bump(cursor.old_no), None)
    body = line[1:] if line.startswith(" ") else line
    return DiffLine(CONTEXT, body, _bump(cursor.old_no), _bump(cursor.new_no))


def render_diff(
    diff: ParsedDiff,
    *,
    max_lines: int | None = None,
    show_line_numbers: bool = True,
) -> Text:
    """Render a parsed diff for a terminal.

    ``max_lines`` caps how many diff lines are emitted; the elision is stated
    in the output rather than silently cut. A failed parse renders as an
    explicit failure message, never as an empty box.
    """
    if not diff.ok:
        reason = diff.error or "the diff could not be rendered"
        return Text(f"diff unavailable: {reason}", style="bold red")

    body = diff.lines
    truncated = max_lines is not None and len(body) > max_lines
    if truncated and max_lines is not None:
        body = body[:max_lines]

    out = Text()
    _append_heading(out, diff)
    gutter_width = _gutter_width(body)
    for line in body:
        style, marker = _STYLE_BY_KIND[line.kind]
        if marker is None:
            out.append(line.text, style=style)
        else:
            _append_body(
                out,
                line,
                marker=marker,
                style=style,
                gutter_width=gutter_width,
                show_line_numbers=show_line_numbers,
            )
        out.append("\n")

    if truncated and max_lines is not None:
        hidden = len(diff.lines) - max_lines
        out.append(
            f"[... {hidden:,} of {len(diff.lines):,} diff lines elided "
            f"(showing the first {max_lines:,}) ...]",
            style="bold yellow",
        )
        out.append("\n")
    return out


def _append_heading(out: Text, diff: ParsedDiff) -> None:
    """Emit the summary/path/count header, omitting whichever parts are absent."""
    heading = diff.summary or diff.path
    if heading:
        out.append(heading, style="bold")
        out.append("\n")
    if diff.added or diff.removed:
        out.append(f"+{diff.added} -{diff.removed}", style="bold")
        out.append("\n")


def _append_body(
    out: Text,
    line: DiffLine,
    *,
    marker: str,
    style: str,
    gutter_width: int,
    show_line_numbers: bool,
) -> None:
    if show_line_numbers:
        gutter = f"{_fmt(line.old_line):>{gutter_width}} {_fmt(line.new_line):>{gutter_width}} "
        out.append(gutter, style="dim")
    out.append(marker, style=style or None)
    out.append(line.text, style=style or None)


def _fmt(number: int | None) -> str:
    return str(number) if number is not None else ""


def _gutter_width(body: list[DiffLine]) -> int:
    highest = max(
        (n for line in body for n in (line.old_line, line.new_line) if n is not None),
        default=0,
    )
    return max(len(str(highest)), 1)


def _bump(number: int | None) -> int | None:
    return None if number is None else number + 1


def _parse_hunk_header(line: str) -> tuple[int | None, int | None]:
    """Return the starting old/new line numbers from a ``@@ -a,b +c,d @@`` header."""
    close = line.find("@@", 3)
    if close < 0:
        return None, None
    range_part = line[3:close].strip()
    parts = range_part.split()
    if len(parts) != 2 or not parts[0].startswith("-") or not parts[1].startswith("+"):
        return None, None
    return _parse_range(parts[0][1:]), _parse_range(parts[1][1:])


def _parse_range(value: str) -> int | None:
    start = value.split(",", 1)[0]
    try:
        return int(start)
    except ValueError:
        return None


def _looks_like_path(head: str) -> bool:
    return bool(head) and "\n" not in head and len(head) <= 255

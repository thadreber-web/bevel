"""Terminal rendering: diffs, markdown, syntax highlighting.

The functions here are pure and take/return plain strings or ``rich``
renderables, so they are unit-testable without a Textual Pilot. Widgets in
:mod:`bevel.tui.widgets` call into this package rather than rendering
themselves.
"""

from bevel.render.diff import DiffLine, DiffLineKind, ParsedDiff, parse_diff, render_diff
from bevel.render.markdown import render_markdown, strip_markdown
from bevel.render.syntax import guess_language, highlight_code

__all__ = [
    "DiffLine",
    "DiffLineKind",
    "ParsedDiff",
    "guess_language",
    "highlight_code",
    "parse_diff",
    "render_diff",
    "render_markdown",
    "strip_markdown",
]

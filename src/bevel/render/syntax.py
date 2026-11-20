"""Syntax highlighting for code blocks and tool output.

pygments is one of the three runtime dependencies and it is here for exactly
this file. Unknown languages degrade to plain text and say so in the returned
text, because a code block that silently loses its highlighting looks like a
code block that lost its content.
"""

from __future__ import annotations

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexer import Lexer
from pygments.lexers import TextLexer, get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound
from rich.text import Text

__all__ = ["guess_language", "highlight_code"]

_STYLE = "monokai"

# Filename suffixes pygments' own filename lookup gets wrong or resolves to a
# poor lexer for the files a coding agent actually touches.
_BY_SUFFIX: dict[str, str] = {
    ".pyi": "python",
    ".toml": "toml",
    ".env": "ini",
}


def guess_language(filename: str) -> str | None:
    """Return a pygments lexer name for ``filename``, or ``None``.

    Only the filename is used. Guessing from content is unreliable for the
    short snippets a tool card shows, and a wrong lexer that colours Python
    like SQL is worse than no colour at all.
    """
    lowered = filename.lower()
    for suffix, language in _BY_SUFFIX.items():
        if lowered.endswith(suffix):
            return language
    try:
        name: str = get_lexer_for_filename(filename).name
    except ClassNotFound:
        return None
    return name


def highlight_code(
    code: str,
    language: str | None = None,
    *,
    filename: str | None = None,
    max_lines: int | None = None,
) -> Text:
    """Highlight ``code``, falling back to plain text with a stated reason.

    ``language`` wins over ``filename``. When neither resolves to a lexer the
    code is returned unhighlighted with a dim note naming what was tried, so
    the reader can tell "no highlighting" from "no content".
    """
    if not code:
        return Text("")

    body = code
    elided: str | None = None
    if max_lines is not None:
        lines = body.splitlines()
        if len(lines) > max_lines:
            body = "\n".join(lines[:max_lines])
            elided = (
                f"[... {len(lines) - max_lines:,} of {len(lines):,} lines elided "
                f"(showing the first {max_lines:,}) ...]"
            )

    resolved = language or (guess_language(filename) if filename else None)
    note: str | None = None
    lexer: Lexer
    if resolved is not None:
        try:
            lexer = get_lexer_by_name(resolved)
        except ClassNotFound:
            lexer = TextLexer()
            note = f"no pygments lexer for {resolved!r}; showing plain text"
    elif filename:
        try:
            lexer = get_lexer_for_filename(filename)
        except ClassNotFound:
            lexer = TextLexer()
            note = f"no pygments lexer for {filename!r}; showing plain text"
    else:
        lexer = TextLexer()

    rendered = highlight(body, lexer, Terminal256Formatter(style=_STYLE, linenos=False))
    text = Text.from_ansi(rendered)
    if note:
        text.append(f"\n{note}", style="dim yellow")
    if elided:
        text.append(f"\n{elided}", style="bold yellow")
    return text

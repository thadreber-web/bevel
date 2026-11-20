"""Markdown rendering for the transcript.

Model output is markdown. This module is the single place that decides how it
looks, so the transcript widget stays a plain function caller and the
behaviour is unit-testable without a Pilot.

Rendering never raises. If markdown cannot be rendered, the caller gets a
visible failure line rather than an empty box that reads as "the model said
nothing".
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text

__all__ = ["render_markdown", "strip_markdown"]


def render_markdown(text: str) -> RenderableType:
    """Render ``text`` as markdown.

    An empty string renders as empty text; there is nothing to draw. A model
    that produced no content is a different thing from a renderer that failed,
    and the caller says which one it has.
    """
    if not text:
        return Text("")
    return Markdown(text)


def strip_markdown(text: str) -> str:
    """Reduce markdown to readable plain text.

    Used where a widget needs a single line (a transcript summary row, a tool
    card title) and a block-level renderer would be wrong. Bold/italic markers,
    inline code ticks, and heading hashes are removed; link targets are kept
    inline as ``label (url)`` so nothing is lost.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        while line.startswith("#"):
            line = line[1:].lstrip()
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = line.replace("*", "").replace("_", "")
        out.append(line)
    return "\n".join(part for part in out if part)

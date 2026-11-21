"""The transcript: everything the model said, in the order it said it.

CONTENT is markdown and goes through :func:`~bevel.render.markdown.render_markdown`.
REASONING is deliberately *not* markdown -- it is dimmed, italic, and collapsed,
so it can never be mistaken for an answer. That distinction is the whole point of
this widget: a transcript that renders thinking like an answer tempts the reader
into trusting prose the model never offered as a result.

Both blocks accumulate rather than render each delta on its own. The loop
streams deltas, and re-rendering the whole block per delta is what keeps
markdown -- which is block-level -- correct without a second incremental parser.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Collapsible, Static

from bevel.render.markdown import render_markdown

#: Title on a reasoning block, visible whether collapsed or not.
REASONING_TITLE = "reasoning"

#: How thinking looks: italic and dim, so it is never mistaken for an answer.
REASONING_STYLE = "italic dim #7f8fa6"


class ContentBlock(Static):
    """One assistant answer, streamed in and rendered as markdown."""

    def __init__(self) -> None:
        self._markdown = ""
        super().__init__(render_markdown(self._markdown))

    def append(self, text: str) -> None:
        self._markdown += text
        self.update(render_markdown(self._markdown))


class ReasoningBlock(Collapsible):
    """One reasoning chunk: dimmed, italic, and folded away by default."""

    def __init__(self) -> None:
        self._text = ""
        self._body = Static(Text("", style=REASONING_STYLE))
        super().__init__(self._body, title=REASONING_TITLE, collapsed=True)

    def append(self, text: str) -> None:
        self._text += text
        self._body.update(Text(self._text, style=REASONING_STYLE))

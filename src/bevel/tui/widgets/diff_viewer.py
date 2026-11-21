"""The diff viewer: a thin, loud wrapper over :mod:`bevel.render.diff`.

Two rules shape this widget, and they are the reason it exists at all rather
than being a call to :func:`~bevel.render.diff.render_diff` at the call site:

* **It never renders an empty box.** A blank region reads as "no changes", which
  is the opposite of the truth whenever the diff failed to arrive or to parse.
  A parse failure, and a diff that parsed to zero lines, both say so in bold red.
* **It does not truncate.** Elision is the render layer's job; it already prints
  how many lines were hidden and what it showed instead.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from bevel.render.diff import ParsedDiff, parse_diff, render_diff

#: Diff lines shown before the render layer elides the rest.
DIFF_LINE_CAP = 200


class DiffView(Static):
    """A unified diff, or a loud explanation of why there isn't one."""

    def __init__(self, diff: str, *, title: str = "diff") -> None:
        super().__init__(_diff_text(diff, title))


def _diff_text(diff: str, title: str) -> Text:
    parsed = parse_diff(diff)
    if parsed.error is not None:
        return Text.assemble(
            Text(f"{title}: could not read the diff", style="bold #f85149"),
            "\n",
            render_diff(parsed),
            "\n",
        )
    if not parsed.lines:
        return Text(f"{title}: the diff contained no lines at all", style="bold #f85149")
    return Text.assemble(
        Text(f"{title} ({len(parsed.lines)} diff lines)", style="bold"),
        "\n",
        _rendered(parsed),
        "\n",
    )


def _rendered(parsed: ParsedDiff) -> Text:
    return render_diff(parsed, max_lines=DIFF_LINE_CAP)


class DiffScreen(ModalScreen[None]):
    """Full-screen diff, dismissed with ``esc``, ``enter`` or ``q``."""

    DEFAULT_CSS = """
    DiffScreen {
        align: center middle;
    }
    DiffScreen > VerticalScroll {
        width: 90%;
        height: 80%;
        padding: 0 1;
        border: round $primary;
        background: #101010;
    }
    """

    def __init__(self, diff: str, *, title: str = "diff") -> None:
        super().__init__()
        self._diff = diff
        self._title = title

    def compose(self) -> ComposeResult:
        with VerticalScroll() as scroll:
            yield DiffView(self._diff, title=self._title)
        scroll.border_title = self._title

    def key_escape(self) -> None:
        self.dismiss(None)

    def key_enter(self) -> None:
        self.dismiss(None)

    def key_q(self) -> None:
        self.dismiss(None)

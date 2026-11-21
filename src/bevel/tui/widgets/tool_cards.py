"""One card per tool execution.

A card answers four questions in order: which tool, what it was asked to do,
how it ended, and what came back. The outcome word is always spelled out and the
title is tinted with the outcome's colour, so colour is never the only channel
carrying the verdict.

Two details are deliberately surfaced rather than dropped:

* ``fixes`` distinguishes a call the repair layer fixed from one that was right
  the first time. A repaired call is a model-quality signal the user should see.
* the output is capped, and the cap is announced. A silently shortened block is
  how a tool that printed an error three hundred lines in comes to look idle.
"""

from __future__ import annotations

from typing import Final

from rich.text import Text
from textual import events
from textual.widgets import Static

from bevel.core.agent import ToolExecution, ToolOutcome
from bevel.render.diff import parse_diff, render_diff
from bevel.tui.widgets.diff_viewer import DiffScreen

#: Output lines shown in a card before the rest is elided.
OUTPUT_LINE_CAP = 200

#: Outcome word and colour per outcome. Every member of ``ToolOutcome`` is here;
#: a new outcome is a KeyError at import time rather than an unstyled card.
_OUTCOMES: Final[dict[ToolOutcome, tuple[str, str]]] = {
    ToolOutcome.OK: ("ok", "#3fb950"),
    ToolOutcome.FAILED: ("failed", "#f85149"),
    ToolOutcome.REJECTED: ("rejected", "#e3b341"),
    ToolOutcome.UNAVAILABLE: ("unavailable", "#bc8cff"),
    ToolOutcome.REPAIR_FAILED: ("repair failed", "#f0883e"),
    ToolOutcome.CANCELLED: ("cancelled", "#8b949e"),
}


class ToolCard(Static):
    """A finished tool call. Focus it and press enter to see the whole diff."""

    can_focus = True

    def __init__(self, execution: ToolExecution) -> None:
        self.execution = execution
        self.diff = _diff_of(execution)
        super().__init__(_card_text(execution))
        self.border_title = execution.name
        if self.diff is None:
            self.can_focus = False

    async def on_key(self, event: events.Key) -> None:
        if event.key in {"enter", "space"} and self.diff is not None:
            event.stop()
            event.prevent_default()
            self.app.push_screen(DiffScreen(self.diff, title=execution_name(self.execution)))


def execution_name(execution: ToolExecution) -> str:
    """The tool's registered name, falling back to what the model asked for."""
    return execution.tool.spec.name if execution.tool is not None else execution.name


def _diff_of(execution: ToolExecution) -> str | None:
    """The execution's output when it is a unified diff, else ``None``."""
    content = execution.result.content if execution.result is not None else ""
    return None if parse_diff(content).error else content


def _card_text(execution: ToolExecution) -> Text:
    word, colour = _OUTCOMES[execution.outcome]
    body = Text()
    body.append(execution_name(execution), style="bold")
    body.append(f" {word}", style=f"bold {colour}")
    if execution.call.arguments:
        body.append(f"  {execution.call.arguments}", style="dim")
    body.append("\n")
    for note in _notes(execution):
        body.append(f"  {note}\n")
    body.append(_output(execution))
    return body


def _notes(execution: ToolExecution) -> list[str]:
    notes: list[str] = []
    if execution.fixes:
        notes.append("repaired: " + "; ".join(execution.fixes))
    notes.extend(f"repair error: {error}" for error in execution.repair_errors)
    if execution.reason:
        notes.append(f"reason: {execution.reason}")
    return notes


def _output(execution: ToolExecution) -> Text:
    """The tool's output, capped, with the elision announced."""
    if execution.result is None:
        return Text("  (no result)", style="dim italic")
    if execution.result.error:
        return Text(f"  {execution.result.error}", style="#f85149")
    content = execution.result.content
    if not content.strip():
        return Text("  (no output)", style="dim italic")
    if _diff_of(execution) is not None:
        return render_diff(parse_diff(content), max_lines=OUTPUT_LINE_CAP)
    lines = content.splitlines()
    shown = lines[:OUTPUT_LINE_CAP]
    text = Text.from_ansi("\n".join(shown))
    text.pad_left(2)
    hidden = len(lines) - len(shown)
    if hidden > 0:
        text.append(
            f"\n  [... {hidden:,} of {len(lines):,} output lines elided "
            f"(showing the first {OUTPUT_LINE_CAP}) ...]",
            style="italic #f0883e",
        )
    return text

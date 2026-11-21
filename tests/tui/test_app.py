"""The TUI driven for real: Textual's Pilot over a scripted provider.

These are not widget-construction tests. Each one runs the actual
:class:`~bevel.tui.app.BevelApp` through ``run_test`` and asserts on what ended
up in the transcript, because the failure mode this suite exists to catch is a
widget that builds fine and then shows the user nothing, or shows them something
that did not come from the loop.

The provider is always :class:`~bevel.providers.mock.MockProvider`. Nothing here
may open a socket.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import pytest
from rich.markdown import Markdown
from rich.text import Text
from textual.command import CommandInput, CommandPalette
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Static

from bevel.config.schema import PolicyConfig, PolicyMode
from bevel.core.agent import UNBUILT_SEAMS, Agent
from bevel.core.checkpoint import snapshot
from bevel.core.messages import Message
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import ContentDelta, GenOptions, ProviderError, StreamEvent
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call
from bevel.tui.app import ApprovalScreen, BevelApp, RunMode
from bevel.tui.widgets.diff_viewer import DiffScreen, DiffView
from bevel.tui.widgets.tool_cards import ToolCard
from bevel.tui.widgets.transcript import REASONING_STYLE, ContentBlock, ReasoningBlock


async def settle(
    pilot: Pilot[None], predicate: Callable[[], bool], *, timeout_s: float = 5.0
) -> bool:
    """Pause until ``predicate`` holds, or until ``timeout_s`` seconds have passed."""
    for _ in range(int(timeout_s / 0.02)):
        await pilot.pause(0.02)
        if predicate():
            return True
    return bool(predicate())


async def submit(pilot: Pilot[None], task: str) -> None:
    """Type a task into the input line and press enter."""
    await pilot.press(*task)
    await pilot.press("enter")


def lines(app: BevelApp) -> list[str]:
    """The transcript as plain strings, in mount order.

    A markdown block reports its source rather than the rendered widget, so a
    test can assert the model's own words arrived intact.
    """
    out: list[str] = []
    for child in app.query_one("#transcript", VerticalScroll).children:
        content = getattr(child, "content", None)
        if isinstance(content, Text):
            out.append(content.plain)
        elif isinstance(content, Markdown):
            out.append(content.markup)
    return out


def styles(app: BevelApp, needle: str) -> list[str]:
    """Every style applied to a span containing ``needle`` in a plain line."""
    found: list[str] = []
    for child in app.query_one("#transcript", VerticalScroll).children:
        content = getattr(child, "content", None)
        if not isinstance(content, Text):
            continue
        for span in content.spans:
            if needle in content.plain[span.start : span.end]:
                found.append(str(span.style))
    return found


# --------------------------------------------------------------------- startup


async def test_seams_are_announced_before_any_task(make_app: Callable[..., BevelApp]) -> None:
    """What is not built is said out loud, on mount, without being asked."""
    app = make_app(ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        await settle(pilot, lambda: len(lines(app)) >= len(UNBUILT_SEAMS))
        shown = lines(app)
    for seam in UNBUILT_SEAMS:
        assert any(seam in line for line in shown), seam
    assert all("not built" in line for line in shown if "m08" in line)


async def test_title_names_the_model_and_the_mode(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(ScriptedTurn(content="hi"), mode=RunMode.PLAN)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        assert app.title == "bevel"
        assert "mock/mock-model" in app.sub_title
        assert "plan mode" in app.sub_title


# ------------------------------------------------------------------- submission


async def test_submitted_task_and_answer_both_render(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(content="**hello** from the model", prompt_tokens=7, completion_tokens=5)
    )
    async with app.run_test() as pilot:
        await submit(pilot, "say hello")
        assert await settle(pilot, lambda: any("done" in line for line in lines(app)))
        shown = lines(app)
    assert any(line.startswith("you  say hello") for line in shown)
    assert any(line == "**hello** from the model" for line in shown), shown
    assert any(line == "done  12 tokens" for line in shown), shown


async def test_content_arrives_as_markdown_not_raw_text(make_app: Callable[..., BevelApp]) -> None:
    """CONTENT goes through the markdown renderer; the source is kept for the test."""
    app = make_app(ScriptedTurn(content="# heading\n\n- one\n- two"))
    async with app.run_test() as pilot:
        await submit(pilot, "list things")
        assert await settle(pilot, lambda: list(app.query(ContentBlock)))
        block = app.query(ContentBlock).first()
        assert isinstance(block.content, Markdown)
        assert block.content.markup == "# heading\n\n- one\n- two"


async def test_empty_answer_says_so(make_app: Callable[..., BevelApp]) -> None:
    """A turn with no content is reported, not rendered as a silent gap."""
    app = make_app(ScriptedTurn(content=""))
    async with app.run_test() as pilot:
        await submit(pilot, "say nothing")
        assert await settle(pilot, lambda: any("no content" in line for line in lines(app)))


async def test_a_second_task_is_refused_while_one_is_running(
    make_app: Callable[..., BevelApp],
) -> None:
    app = make_app(
        ScriptedTurn(content="working", tool_calls=[tool_call("c", "glob", {"pattern": "*"})]),
        delay=0.4,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "first")
        assert await settle(pilot, lambda: app._agent is not None)
        await submit(pilot, "second")
        assert await settle(pilot, lambda: any("already going" in line for line in lines(app)))
        await pilot.press("ctrl+c")
        assert await settle(pilot, lambda: app._agent is None)


# --------------------------------------------------------------------- tooling


async def test_tool_call_renders_a_card_coloured_by_its_outcome(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "read", {"path": "a.py"})]),
        ScriptedTurn(content="read it"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "read a.py")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.border_title == "read"
        assert card.content.plain.startswith("read ok")
        assert "1|one" in card.content.plain
        assert str(card.content.spans[1].style) == "bold #3fb950"
        assert card.diff is None
        assert card.can_focus is False


async def test_failed_tool_shows_its_error_in_red(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "read", {"path": "missing.txt"})]),
        ScriptedTurn(content="sorry"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "read a missing file")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.content.plain.startswith("read failed")
        assert "no such file: missing.txt" in card.content.plain
        assert any("f85149" in style for style in _card_styles(card))


async def test_unknown_tool_is_reported_as_unavailable(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "teleport", {})]),
        ScriptedTurn(content="nope"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "teleport")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.content.plain.startswith("teleport unavailable")
        assert "reason:" in card.content.plain


async def test_a_repaired_call_says_it_was_repaired(make_app: Callable[..., BevelApp]) -> None:
    """``fixes`` is the difference between a fixed call and a first-time-right one."""
    app = make_app(
        ScriptedTurn(
            tool_calls=[tool_call("c1", "read", {"path": "a.py", "offset": "1"})],
        ),
        ScriptedTurn(content="ok"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "read with a stringy offset")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert "repaired:" in card.content.plain


async def test_long_output_is_capped_and_the_cap_is_announced(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    (tmp_path / "big.txt").write_text("\n".join(f"line {n}" for n in range(500)), encoding="utf-8")
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "read", {"path": "big.txt", "limit": 500})]),
        ScriptedTurn(content="done"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "read the big file")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        plain = app.query(ToolCard).first().content.plain
    body = [line for line in plain.splitlines() if line.strip().startswith("499|")]
    assert not body, "output past the cap leaked into the card"
    assert "of 500 output lines elided (showing the first 200)" in plain


# ------------------------------------------------------------------------ diff


async def test_an_edit_renders_its_hunks_and_opens_full_screen(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    app = make_app(
        ScriptedTurn(
            tool_calls=[
                tool_call(
                    "c1",
                    "edit",
                    {"path": "a.py", "edits": [{"old_text": "two", "new_text": "TWO"}]},
                )
            ]
        ),
        ScriptedTurn(content="edited"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "change two to TWO")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.diff is not None
        assert card.can_focus is True
        plain = card.content.plain
        assert "a.py: 2 lines changed" in plain
        assert "@@ -1,3 +1,3 @@" in plain
        assert "-two" in plain
        assert "+TWO" in plain

        card.focus()
        await pilot.pause(0.05)
        await pilot.press("enter")
        assert await settle(pilot, lambda: isinstance(app.screen, DiffScreen))
        view = app.screen.query(DiffView).first()
        assert "a.py" in view.content.plain
        assert "+TWO" in view.content.plain
        await pilot.press("escape")
        assert await settle(pilot, lambda: not isinstance(app.screen, DiffScreen))


async def test_a_non_diff_output_never_opens_a_screen(make_app: Callable[..., BevelApp]) -> None:
    """A card with no diff is not focusable, so enter cannot open an empty viewer."""
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "read", {"path": "conftest.py"})]),
        ScriptedTurn(content="ok"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "read something")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.can_focus is False
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert not any(isinstance(s, DiffScreen) for s in app.screen_stack)


# ------------------------------------------------------------------- reasoning


async def test_reasoning_is_collapsed_dimmed_and_italic(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(ScriptedTurn(content="the answer", reasoning="a long chain of thought"))
    async with app.run_test() as pilot:
        await submit(pilot, "think about it")
        assert await settle(pilot, lambda: list(app.query(ReasoningBlock)))
        block = app.query(ReasoningBlock).first()
        assert block.collapsed is True
        assert str(block.title) == "reasoning"
        body = [child for child in block.walk_children() if isinstance(child, Static)]
        text = next(c.content for c in body if isinstance(c.content, Text))
        assert text.plain == "a long chain of thought"
        assert str(text.style) == REASONING_STYLE
        # Distinct from content: the answer is markdown, the thinking is not.
        assert not isinstance(app.query(ContentBlock).first().content, Text)


async def test_reasoning_and_content_keep_their_order(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(content="first", reasoning="before"),
        ScriptedTurn(content="second", reasoning="after"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "two turns")
        assert await settle(pilot, lambda: any("done" in line for line in lines(app)))
        kinds = [type(c).__name__ for c in app.query_one("#transcript").children]
        assert kinds.index("ReasoningBlock") < kinds.index("ContentBlock")


# ----------------------------------------------------------------- interruptions


class _Boom(MockProvider):
    """Streams a chunk, then the connection dies."""

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(
            {"messages": list(messages), "tools": list(tools or []), "options": options}
        )
        return self._explode()

    async def _explode(self) -> AsyncIterator[StreamEvent]:
        yield ContentDelta("partial answer")
        raise ProviderError("connection reset by peer")


async def test_a_mid_stream_provider_error_surfaces_visibly(
    tmp_path: Path, isolated_homes: None
) -> None:
    """The specific thing this guards: an exception must not end the run silently."""

    app = BevelApp(
        provider=_Boom(turns=[]),
        workspace=tmp_path,
        policy=PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db"),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "boom")
        assert await settle(pilot, lambda: any("provider error" in line for line in lines(app)))
        assert any("partial answer" in line for line in lines(app))
        assert app._agent is None
        assert any("f85149" in style for style in styles(app, "provider error"))


async def test_ctrl_c_cancels_the_agent_and_ends_the_run(
    make_app: Callable[..., BevelApp], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just that a cancelled event appeared -- that ``cancel()`` was reached."""
    calls: list[tuple[Agent, str]] = []
    original = Agent.cancel

    def spy(self: Agent, reason: str = "cancelled by user") -> None:
        calls.append((self, reason))
        original(self, reason)

    monkeypatch.setattr(Agent, "cancel", spy)
    app = make_app(
        ScriptedTurn(content="looping", tool_calls=[tool_call("c", "glob", {"pattern": "*"})]),
        delay=0.05,
    )
    provider = app.provider
    assert isinstance(provider, MockProvider)
    async with app.run_test() as pilot:
        await submit(pilot, "loop forever")
        assert await settle(pilot, lambda: app._agent is not None)
        await pilot.press("ctrl+c")
        assert await settle(pilot, lambda: app._agent is None)
        assert calls, "Agent.cancel was never called"
        assert calls[0][1] == "cancelled by user"
        assert any("cancelled  cancelled by user" in line for line in lines(app))
        # Counted only once the run is over: a request already in flight when
        # Ctrl+C lands is legitimate and closes normally, so comparing against a
        # count taken mid-run would fail on correct behaviour.
        asked = len(provider.requests)
        assert asked > 0
        await pilot.pause(0.3)
        assert len(provider.requests) == asked, "the loop kept asking after the run ended"


async def test_ctrl_c_quits_when_nothing_is_running(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.press("ctrl+c")
        assert await settle(pilot, lambda: app._exit)


async def test_resizing_mid_run_keeps_the_transcript(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(
            content="streaming along", tool_calls=[tool_call("c", "glob", {"pattern": "*"})]
        ),
        delay=0.05,
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await submit(pilot, "keep going")
        assert await settle(pilot, lambda: app._agent is not None)
        await pilot.resize_terminal(50, 15)
        await pilot.pause(0.1)
        assert app.size.width == 50
        assert app.size.height == 15
        assert any("streaming along" in line for line in lines(app))
        await pilot.press("ctrl+c")
        assert await settle(pilot, lambda: app._agent is None)
        await pilot.resize_terminal(100, 40)
        await pilot.pause(0.1)
        assert app.size.width == 100


# ------------------------------------------------------------------------ modes


async def test_plan_mode_refuses_a_write_and_says_why(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "write", {"path": "x.txt", "content": "hi"})]),
        ScriptedTurn(content="understood"),
        mode=RunMode.PLAN,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "make a file")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        card = app.query(ToolCard).first()
        assert card.content.plain.startswith("write rejected")
        assert "plan mode is read-only" in card.content.plain
        assert not (tmp_path / "x.txt").exists()


async def test_default_mode_asks_and_a_refusal_blocks_the_write(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "write", {"path": "x.txt", "content": "hi"})]),
        ScriptedTurn(content="understood"),
        mode=RunMode.DEFAULT,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "make a file")
        assert await settle(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        assert "write" in app.screen.request.name
        await pilot.press("n")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        assert app.query(ToolCard).first().content.plain.startswith("write rejected")
        assert not (tmp_path / "x.txt").exists()


async def test_default_mode_allows_on_y(make_app: Callable[..., BevelApp], tmp_path: Path) -> None:
    app = make_app(
        ScriptedTurn(tool_calls=[tool_call("c1", "write", {"path": "x.txt", "content": "hi"})]),
        ScriptedTurn(content="done"),
        mode=RunMode.DEFAULT,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "make a file")
        assert await settle(pilot, lambda: isinstance(app.screen, ApprovalScreen))
        await pilot.press("y")
        assert await settle(pilot, lambda: list(app.query(ToolCard)))
        assert app.query(ToolCard).first().content.plain.startswith("write ok")
        assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "hi"


async def test_f2_cycles_the_mode_and_the_hint_travels_with_it(
    make_app: Callable[..., BevelApp],
) -> None:
    app = make_app(ScriptedTurn(content="hi"), mode=RunMode.DEFAULT)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.press("f2")
        assert await settle(pilot, lambda: app.mode is RunMode.AUTO)
        assert any("workspace writes are free" in line for line in lines(app))
        await pilot.press("f2")
        assert await settle(pilot, lambda: app.mode is RunMode.PLAN)
        assert "plan mode" in app.sub_title


async def test_mode_cannot_be_changed_under_a_run(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(
        ScriptedTurn(content="working", tool_calls=[tool_call("c", "glob", {"pattern": "*"})]),
        delay=0.4,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "go")
        assert await settle(pilot, lambda: app._agent is not None)
        before = app.mode
        await pilot.press("f2")
        await pilot.pause(0.1)
        assert app.mode is before
        assert any("Ctrl+C to stop it" in line for line in lines(app))
        await pilot.press("ctrl+c")
        assert await settle(pilot, lambda: app._agent is None)


# ------------------------------------------------------------------------ undo


async def test_undo_restores_the_last_checkpoint(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    """Only meaningful over a real repository, so the workspace is one.

    The checkpoint captures the tree at the moment it is taken, so the change
    that undo has to undo is made *after* the snapshot.
    """

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=bevel", "-c", "user.email=b@l", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    git("add", "a.py")
    git("commit", "-qm", "base")
    assert (await snapshot(tmp_path, session="s1")).ok
    (tmp_path / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")

    app = make_app(ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await app.undo_last_run()
        await pilot.pause(0.2)
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "one\n"
        assert any(line.startswith("undo  restored s1") for line in lines(app))


async def test_undo_without_checkpoints_says_so(make_app: Callable[..., BevelApp]) -> None:
    app = make_app(ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await app.undo_last_run()
        await pilot.pause(0.2)
        assert any("no checkpoints" in line for line in lines(app))


# -------------------------------------------------------------------- palette


async def test_the_palette_runs_a_command(make_app: Callable[..., BevelApp]) -> None:
    """f1 opens the palette and picking a command really runs it."""
    app = make_app(ScriptedTurn(content="hi"), mode=RunMode.DEFAULT)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.press("f1")
        assert await settle(pilot, lambda: isinstance(app.screen, CommandPalette))
        await pilot.press(*"mode auto")
        assert await settle(
            pilot, lambda: app.screen.query(CommandInput).first().value == "mode auto"
        )
        await pilot.press("enter")
        assert await settle(pilot, lambda: app.mode is RunMode.AUTO)
        assert not isinstance(app.screen, CommandPalette)
        assert any("workspace writes are free" in line for line in lines(app))


def _card_styles(card: ToolCard) -> list[str]:
    return [str(span.style) for span in card.content.spans]

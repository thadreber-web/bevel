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

from bevel.config import loader
from bevel.config.schema import MemoryConfig, PolicyConfig, PolicyMode
from bevel.core.agent import UNBUILT_SEAMS, Agent, ToolOutcome
from bevel.core.checkpoint import snapshot
from bevel.core.memory.store import MemoryStore, MemoryType, repo_scope
from bevel.core.messages import Message, ToolCall
from bevel.core.session.resume import rebuild_transcript
from bevel.core.session.store import SessionStore, TodoStore
from bevel.core.tools.base import ToolResult
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import Capabilities, ContentDelta, GenOptions, ProviderError, StreamEvent
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call
from bevel.tui.app import ApprovalScreen, BevelApp, RunMode, run_tui
from bevel.tui.widgets.diff_viewer import DiffScreen, DiffView
from bevel.tui.widgets.tool_cards import ToolCard
from bevel.tui.widgets.transcript import REASONING_STYLE, ContentBlock, ReasoningBlock

#: Shared by the memory tests. They share no words with each other or with the
#: task, so a store that returned everything would fail the second assertion.
MATCHED_MEMORY = "the auth middleware broke on empty tokens"
UNMATCHED_MEMORY = "the caching layer indexes documents nightly"


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
    """What is not built is said out loud, on mount, without being asked.

    m08c deliberately flipped this assertion: session persistence used to be
    announced here as unbuilt. It is built now, so the only seam left in m08 is
    the L3 handoff, and a run must say so without also claiming it cannot store.
    """
    app = make_app(ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        await settle(pilot, lambda: len(lines(app)) >= len(UNBUILT_SEAMS))
        shown = lines(app)
    assert len(UNBUILT_SEAMS) == 1
    for seam in UNBUILT_SEAMS:
        assert any(seam in line for line in shown), seam
    assert all("not built" in line for line in shown if "m08" in line)
    assert not any("not built" in line and "persist" in line for line in shown)


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


async def test_greeting_gets_a_reply_and_no_tool_cards(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    """The manual check, automated: "hello" is answered, and nothing was run first.

    The provider is scripted to answer without tools, so the assertion that earns
    its keep here is that the project map was already on the wire -- the
    orientation that makes a tool-free greeting the model's obvious move.
    """
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    app = make_app(ScriptedTurn(content="Hi. What are we doing?"))
    async with app.run_test() as pilot:
        await submit(pilot, "hello")
        assert await settle(pilot, lambda: any("done" in line for line in lines(app)))
        shown = lines(app)
        sent = app.provider.requests[0]["messages"]
    assert any(line == "Hi. What are we doing?" for line in shown), shown
    assert not list(app.query(ToolCard)), "a greeting must not cost the user tool cards"
    text = "\n".join(message.content for message in sent)
    assert "# project map:" in text
    assert 'name = "demo"' in text


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
        # The reason, not just the word. "provider error" alone cannot tell a
        # certificate you must trust from a model name the server lacks.
        assert any("connection reset by peer" in line for line in lines(app)), (
            "the provider error was reported without its reason"
        )


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


# ------------------------------------------------------------------- compaction


def _context_lines(app: BevelApp) -> list[str]:
    """The transcript's compaction lines, which the app labels ``context``."""
    return [line for line in lines(app) if line.startswith("context")]


def _context_app(tmp_path: Path, *turns: ScriptedTurn, window: int | None = 100) -> BevelApp:
    """An app whose provider reports a deliberately tiny context window.

    The loop trusts the server's own ``prompt_tokens`` against the server's own
    ``max_model_len``, so scripting both is the whole apparatus -- no tokeniser,
    no endpoint.
    """
    provider = MockProvider(turns=list(turns))
    provider.capabilities = Capabilities(max_model_len=window, supports_tools=True)
    policy = PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db")
    return BevelApp(provider=provider, workspace=tmp_path, policy=policy, mode=RunMode.AUTO)


async def test_a_long_run_shows_a_context_line_naming_its_level(tmp_path: Path) -> None:
    """The transcript says *which* level ran and *how much* it removed.

    A line that just said "context compacted" would be the UI claiming work the
    loop may not have done, so the level and the number have to be visible.
    """
    (tmp_path / "big.txt").write_text(
        "\n".join(f"line {n} {'x' * 60}" for n in range(400)), encoding="utf-8"
    )
    turns = [
        ScriptedTurn(
            content=f"looking at {n} " + "y" * 300,
            tool_calls=[tool_call(f"c{n}", "read", {"path": "big.txt", "limit": 400})],
            prompt_tokens=0,
        )
        for n in range(1, 7)
    ]
    turns[5] = ScriptedTurn(
        content="looking at 6 " + "y" * 300,
        tool_calls=[tool_call("c6", "read", {"path": "big.txt", "limit": 400})],
        prompt_tokens=900,
    )
    turns.append(ScriptedTurn(content="done", prompt_tokens=900))
    app = _context_app(tmp_path, *turns)

    async with app.run_test() as pilot:
        await submit(pilot, "read the big file a lot")
        assert await settle(pilot, lambda: _context_lines(app), timeout_s=20.0)
        context = _context_lines(app)

    assert context, "no context line was written"
    assert any(("l1:" in line or "l2:" in line) for line in context), context
    assert any("chars" in line for line in context), context


async def test_the_l3_handoff_is_announced_as_not_built_on_mount(tmp_path: Path) -> None:
    """Mount says the L3 handoff does not exist, before anyone asks."""
    app = _context_app(tmp_path, ScriptedTurn(content="hi"))
    async with app.run_test() as pilot:
        assert await settle(pilot, lambda: len(lines(app)) >= len(UNBUILT_SEAMS))
        shown = lines(app)

    l3 = [line for line in shown if "l3" in line.lower()]
    assert l3, shown
    assert all("not built" in line for line in l3)
    assert any("context_overflow" in line for line in l3)


async def test_a_context_overflow_stops_the_run_with_a_readable_line(tmp_path: Path) -> None:
    """The stop line is the user's only notice that the run was cut short.

    One turn with one short observation: L1 has nothing to cut and there is no
    middle for L2 to summarise, so both levels are exhausted on the first
    over-budget measurement and the run must stop rather than be sent.
    """
    (tmp_path / "small.txt").write_text("one\ntwo\n", encoding="utf-8")
    app = _context_app(
        tmp_path,
        ScriptedTurn(
            content="thinking " + "y" * 2000,
            tool_calls=[tool_call("c1", "read", {"path": "small.txt"})],
            prompt_tokens=900,
        ),
    )
    async with app.run_test() as pilot:
        await submit(pilot, "do something long")
        assert await settle(
            pilot, lambda: any("would not fit" in line for line in lines(app)), timeout_s=20.0
        )
        shown = lines(app)

    assert any("stopped: context would not fit" in line for line in shown), shown
    assert any("L3" in line for line in shown), shown


# ---------------------------------------------------------------------- resume


async def test_a_resumed_app_shows_the_transcript_from_the_store(
    make_app: Callable[..., BevelApp],
) -> None:
    """The reopened transcript is read back from SQLite, not held in memory.

    The proof is that the second app is a *different process-local object* with
    no memory of the first run: everything it displays has to come off the disk.
    """
    first = make_app(
        ScriptedTurn(
            content="the passphrase is zephyr-quartz",
            tool_calls=[tool_call("c1", "read", {"path": "notes.txt"})],
        ),
        ScriptedTurn(content="done"),
    )
    (first.workspace / "notes.txt").write_text("secret notes", encoding="utf-8")
    async with first.run_test() as pilot:
        await submit(pilot, "read my notes and tell me the passphrase")
        assert await settle(pilot, lambda: any("done" in line for line in lines(first)))
    assert first.resume is not None, "the run did not leave a session behind"
    assert first.store is not None
    store = first.store
    assert store is not None

    reopened = rebuild_transcript(SessionStore(store.db_path), first.resume.session.id)
    assert reopened is not None
    app = make_app(ScriptedTurn(content="zephyr-quartz"), resume=reopened)
    async with app.run_test() as pilot:
        assert await settle(pilot, lambda: any("resumed" in line for line in lines(app)))
        shown = lines(app)
        cards = list(app.query(ToolCard))

    assert any("read my notes and tell me the passphrase" in line for line in shown), shown
    assert any("the passphrase is zephyr-quartz" in line for line in shown), shown
    assert len(cards) == 1
    assert cards[0].execution.name == "read"
    assert cards[0].execution.outcome is ToolOutcome.OK
    assert "secret notes" in (cards[0].execution.result or ToolResult(ok=False)).content


async def test_a_resumed_app_sends_the_stored_history_to_the_model(
    make_app: Callable[..., BevelApp],
) -> None:
    """What the user sees is not the point; what the endpoint receives is."""
    first = make_app(ScriptedTurn(content="the capital is Llanfairpwll"))
    async with first.run_test() as pilot:
        await submit(pilot, "what is the capital?")
        assert await settle(pilot, lambda: any("Llanfairpwll" in line for line in lines(first)))
    assert first.resume is not None
    assert first.store is not None

    reopened = rebuild_transcript(SessionStore(first.store.db_path), first.resume.session.id)
    assert reopened is not None
    second = make_app(ScriptedTurn(content="Llanfairpwll, again"), resume=reopened)
    async with second.run_test() as pilot:
        await submit(pilot, "say it once more")
        assert await settle(pilot, lambda: any("again" in line for line in lines(second)))

    sent = second.provider.requests[0]["messages"]
    assert [message.role for message in sent] == ["system", "user", "assistant", "user"]
    assert sent[1].content == "what is the capital?"
    assert sent[3].content == "say it once more"


async def test_a_resumed_app_announces_what_it_had_to_drop(
    make_app: Callable[..., BevelApp], tmp_path: Path
) -> None:
    """A session whose last run died mid-batch says so instead of hiding it."""
    store = SessionStore(loader.session_db_path())
    info = store.create_session(workspace=tmp_path, provider="mock", model="mock-model")
    store.append_turn(info.id, Message.user("read both"), iteration=1)
    store.append_turn(
        info.id,
        Message(
            "assistant",
            tool_calls=(ToolCall("c1", "read", "{}"), ToolCall("c2", "read", "{}")),
        ),
        iteration=1,
    )
    store.append_turn(info.id, Message.tool_result("c1", "the first half"), iteration=1)

    reopened = rebuild_transcript(store, info.id)
    assert reopened is not None
    app = make_app(ScriptedTurn(content="ok"), resume=reopened)
    async with app.run_test() as pilot:
        assert await settle(pilot, lambda: any("dropped" in line for line in lines(app)))
        shown = lines(app)

    assert any("dropped" in line and "2 trailing message" in line for line in shown), shown
    assert any(line.startswith("you") and "read both" in line for line in shown), shown
    assert not any("the first half" in line for line in shown), shown


async def test_a_run_without_a_store_still_works(make_app: Callable[..., BevelApp]) -> None:
    """Persistence is attached, not required; the TUI must not need a database."""
    app = make_app(ScriptedTurn(content="no database needed"), store=False)
    async with app.run_test() as pilot:
        await submit(pilot, "are you storing anything")
        assert await settle(pilot, lambda: any("no database needed" in line for line in lines(app)))
    assert not loader.session_db_path().exists()


# ----------------------------------------------------------------------- memory


async def test_the_tui_injects_recalled_memory_into_the_first_request(
    tmp_path: Path, isolated_homes: None
) -> None:
    """`[memory] enabled = true` has to reach the loop through the TUI.

    The agent already takes ``memory=`` and the tools are always offered, so a
    store that is built but never handed over is the silent fake: config says
    memory is on, and the model is shown nothing. The assertion is on the
    request the provider actually received, not on a widget.

    The relevant memory is filed under the *workspace* scope rather than
    ``global``, so this also catches injection reading from the wrong tier: the
    global tier is always searched, so a global-only fixture passes even when
    the workspace scope is never looked at.
    """
    memory = MemoryStore(tmp_path / "memories.db", config=MemoryConfig(enabled=True))
    memory.add(content=MATCHED_MEMORY, type=MemoryType.FACT, scope=await repo_scope(tmp_path))
    memory.add(content=UNMATCHED_MEMORY, type=MemoryType.FACT, scope="global")

    provider = MockProvider(turns=[ScriptedTurn(content="noted")])
    app = BevelApp(
        provider=provider,
        workspace=tmp_path,
        policy=PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db"),
        mode=RunMode.AUTO,
        memory=memory,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "how does bevel handle auth middleware")
        assert await settle(pilot, lambda: any("noted" in line for line in lines(app)))

    sent = provider.requests[0]["messages"]
    text = "\n".join(message.content for message in sent)
    assert MATCHED_MEMORY in text, "the TUI never passed the memory store to the loop"
    assert UNMATCHED_MEMORY not in text, "an irrelevant memory reached the model"
    memory.close()


def test_run_tui_only_builds_a_store_when_config_turns_memory_on(
    tmp_path: Path,
    isolated_homes: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[memory] enabled`` has to be what decides, not nothing at all.

    Handing a store to ``BevelApp`` by hand, as the test above does, proves the
    app passes it down. It says nothing about whether ``run_tui`` ever built one
    from config -- which is precisely the line that was missing, and the reason
    the feature was inert for a user who turned it on. So this drives ``run_tui``
    itself with the app stubbed out, and reads back what the entry point decided.

    Sync, not async: ``run_tui`` owns the event loop, and ``asyncio.run`` cannot
    be called from inside one.
    """
    for name in ("BEVEL_BASE_URL", "BEVEL_MODEL", "BEVEL_PROFILE", "BEVEL_CONFIG"):
        monkeypatch.delenv(name, raising=False)

    captured: dict[str, object] = {}

    class RecordingApp:
        """Stands in for the app so no alternate screen is ever opened."""

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.__dict__.update(kwargs)

        async def run_async(self) -> int:
            return 0

    monkeypatch.setattr("bevel.tui.app.BevelApp", RecordingApp)

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    def run(enabled: bool) -> object:
        (config_dir / "config.toml").write_text(
            "[profile.home]\n"
            'base_url = "https://models.test/v1"\n'
            'model = "test-model"\n'
            f"\n[memory]\nenabled = {'true' if enabled else 'false'}\n",
            encoding="utf-8",
        )
        captured.clear()
        assert run_tui(workspace=tmp_path) == 0
        return captured["memory"]

    on = run(True)
    assert isinstance(on, MemoryStore), "memory is on in config but the loop got no store"
    assert on.config.enabled is True
    assert on.db_path == loader.memory_db_path()
    on.close()

    assert run(False) is None, "memory is off but a store reached the loop anyway"


async def test_the_tui_without_a_memory_store_never_mentions_recall(
    make_app: Callable[..., BevelApp],
) -> None:
    """Memory off must be the pre-m09a TUI byte-for-byte, not an empty block."""
    app = make_app(ScriptedTurn(content="plain"))
    async with app.run_test() as pilot:
        await submit(pilot, "anything from memory")
        assert await settle(pilot, lambda: any("plain" in line for line in lines(app)))

    sent = app.provider.requests[0]["messages"]
    assert all("bevel recall" not in message.content for message in sent)


# ------------------------------------------------------------------------ todos


async def test_the_tui_writes_a_scripted_todo_list_to_the_database(
    tmp_path: Path, isolated_homes: None
) -> None:
    """The whole user path: type a task, the model plans, the plan is on disk.

    ``run_tui`` builds the todo store and hands it to the app, the app hands it
    to the loop, and the loop assigns it the session id. Any one of those three
    links going missing leaves the user with a tool that reports success and a
    ``bevel todos`` that shows nothing, so the assertion is on the reopened
    database rather than on the transcript.
    """
    call = tool_call(
        "c1",
        "todo",
        {
            "todos": [
                {"content": "read the parser", "status": "done"},
                {"content": "fix it", "status": "pending"},
            ]
        },
    )
    provider = MockProvider(
        turns=[ScriptedTurn(content="", tool_calls=(call,)), ScriptedTurn(content="planned")]
    )
    store = SessionStore(loader.session_db_path())
    todos = TodoStore(store.db_path)
    app = BevelApp(
        provider=provider,
        workspace=tmp_path,
        policy=PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db"),
        mode=RunMode.AUTO,
        store=store,
        todos=todos,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "fix the parser")
        assert await settle(pilot, lambda: any("planned" in line for line in lines(app)))

    assert app.store is not None
    assert todos.session_id is not None, (
        "the loop never told the todo store which session it was writing to"
    )
    reopened = TodoStore(store.db_path)
    reopened.session_id = todos.session_id
    assert [(t.content, str(t.status)) for t in reopened.list_todos()] == [
        ("read the parser", "done"),
        ("fix it", "pending"),
    ], "the TUI never wired the todo store through to the loop"
    reopened.close()
    store.close()
    todos.close()


async def test_the_tui_without_a_todo_store_leaves_the_prompt_silent_on_todos(
    tmp_path: Path, isolated_homes: None
) -> None:
    """No store means no guidance, so the model is not pushed at a tool that fails."""
    provider = MockProvider(turns=[ScriptedTurn(content="plain")])
    app = BevelApp(
        provider=provider,
        workspace=tmp_path,
        policy=PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db"),
        mode=RunMode.AUTO,
    )
    async with app.run_test() as pilot:
        await submit(pilot, "anything")
        assert await settle(pilot, lambda: any("plain" in line for line in lines(app)))

    sent = app.provider.requests[0]["messages"]
    text = "\n".join(message.content for message in sent).lower()
    # Not the bare word "todo": the project map lists ``tools/todo.py`` whether
    # or not the feature is on. What must be absent is the guidance block.
    assert "task tracking is on" not in text


def test_run_tui_builds_a_todo_store_on_the_session_database_and_closes_it(
    tmp_path: Path,
    isolated_homes: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_tui`` owns the store's whole lifetime, so that is where this is proven.

    Handing a store to ``BevelApp`` by hand proves the app passes it down; it
    says nothing about the entry point ever building one, or about it being
    closed when the app exits. A leaked connection is the kind of thing that
    only shows up as a locked file on Windows.
    """
    for name in ("BEVEL_BASE_URL", "BEVEL_MODEL", "BEVEL_PROFILE", "BEVEL_CONFIG"):
        monkeypatch.delenv(name, raising=False)

    captured: dict[str, object] = {}

    class RecordingApp:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.__dict__.update(kwargs)

        async def run_async(self) -> int:
            return 0

    monkeypatch.setattr("bevel.tui.app.BevelApp", RecordingApp)
    closed: list[TodoStore] = []
    real_close = TodoStore.close
    monkeypatch.setattr(TodoStore, "close", lambda self: (closed.append(self), real_close(self))[1])
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        '[profile.home]\nbase_url = "https://models.test/v1"\nmodel = "test-model"\n',
        encoding="utf-8",
    )

    assert run_tui(workspace=tmp_path) == 0

    todos = captured["todos"]
    assert isinstance(todos, TodoStore), "the TUI never built a todo store"
    assert todos.db_path == loader.session_db_path(), (
        "todos live in sessions.db, not somewhere else"
    )
    assert closed == [todos], "the entry point left the todo store open"

"""L1 and L2 compaction, and the loop step that runs them.

Step 8 of the loop used to be a stub: it emitted a COMPACTION event admitting that
"nothing compacted". These tests pin the two levels that exist now, and pin the
honesty of the two that do not — a COMPACTION event that says "compacted" without a
number is a fake, and a run that silently goes unoriented after compaction is the
worst failure mode available here.

Everything is deterministic and needs no endpoint. The trigger is the server's own
number, so ``ScriptedTurn(prompt_tokens=...)`` against
``Capabilities(max_model_len=...)`` is the whole apparatus. Nothing in the loop
counts tokens, and ``MockProvider.count_tokens`` is deliberately never consulted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bevel.core.agent import (
    UNBUILT_SEAMS,
    Agent,
    AgentEvent,
    AgentEventType,
    Budgets,
    StopReason,
)
from bevel.core.context.compact import (
    _SUMMARY_PREFIX,
    HEAD_CHARS,
    KEEP_RECENT_TURNS,
    MIN_OBSERVATION_CHARS,
    TAIL_CHARS,
    compact_l1,
    compact_l2,
)
from bevel.core.messages import Message
from bevel.core.policy import allow_all_policy
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import AssistantTurn, Capabilities, GenOptions, Usage
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

LONG_OBSERVATION = "OBSERVATION\n" + "\n".join(
    f"observation line {i} {'x' * 40}" for i in range(120)
)
"""Well over MIN_OBSERVATION_CHARS, with lines long enough that head+tail is a small
fraction of the whole."""

OVER = 900
WITHIN = 50
WINDOW = 100
"""Scripted ``prompt_tokens`` against a window of 100: OVER trips the trigger, WITHIN
does not. The loop reads only the server's number, so these are the whole apparatus."""


class RecordingSummariser:
    """A provider that only ever summarises, and remembers what it was shown."""

    name = "summariser"
    model = "summariser-model"

    def __init__(self, summary: str = "they read six files") -> None:
        self.summary = summary
        self.shown: list[list[Message]] = []
        self.capabilities = Capabilities()

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,  # noqa: ARG002
        options: GenOptions | None = None,  # noqa: ARG002
    ) -> AssistantTurn:
        self.shown.append(list(messages))
        return AssistantTurn(
            message=Message("assistant", self.summary),
            finish_reason="stop",
            usage=Usage(prompt_tokens=500, completion_tokens=20, total_tokens=520),
            model=self.model,
        )

    async def summarise(self, middle: list[Message]) -> str:
        """The ``Summarise`` shape, for calling ``compact_l2`` directly."""

        turn = await self.complete(middle)
        return turn.message.content

    async def aclose(self) -> None:
        return None


class FetchTool:
    """A read tool whose observation is long and fixed, so L1 has something to cut."""

    is_parallel_safe = True

    def __init__(self, observation: str = LONG_OBSERVATION) -> None:
        self.risk_tier = RiskTier.READ
        self.spec = ToolSpec(
            name="fetch",
            description="Fetch the contents of a path.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        self.observation = observation
        self.calls = 0

    async def execute(self, arguments: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        del arguments, ctx
        self.calls += 1
        return ToolResult(ok=True, content=self.observation)


def _agent(
    turns: list[ScriptedTurn],
    tmp_path: Path,
    *,
    window: int | None = WINDOW,
    tools: list[Any] | None = None,
    summariser: Any = None,
) -> Agent:
    provider = MockProvider(turns=turns)
    provider.capabilities = Capabilities(
        max_model_len=window, supports_tools=True, supports_streaming=True
    )
    return Agent(
        provider,
        tmp_path,
        tools=tools or [],
        budgets=Budgets(max_iterations=12, max_tool_calls=40),
        policy=allow_all_policy,
        summariser=summariser,
    )


def _fetch_turn(call_id: str = "c1", prompt_tokens: int = OVER) -> ScriptedTurn:
    return ScriptedTurn(
        content="",
        tool_calls=(tool_call(call_id, "fetch", {"path": "notes.txt"}),),
        prompt_tokens=prompt_tokens,
    )


def _final(prompt_tokens: int = OVER) -> ScriptedTurn:
    return ScriptedTurn(content="done", prompt_tokens=prompt_tokens)


def _long_run() -> list[ScriptedTurn]:
    """Six prose-plus-tool turns then a final turn, over budget only on the sixth.

    The earlier turns report no ``prompt_tokens``, so the loop honestly knows nothing
    about its context size while the transcript is being built. The sixth is the
    first measurement, and by then there are more assistant turns than L2 keeps, so
    L2 has a middle to replace. The prose is what makes the middle worth replacing:
    L1 only cuts tool observations, so without it L2 would have nothing to gain.
    """
    turns = [
        ScriptedTurn(
            content=f"thinking about file {i} " + "y" * 300,
            tool_calls=(tool_call(f"c{i}", "fetch", {"path": f"notes{i}.txt"}),),
            prompt_tokens=0,
        )
        for i in range(1, 7)
    ]
    turns[5] = ScriptedTurn(
        content="thinking about file 6 " + "y" * 300,
        tool_calls=(tool_call("c6", "fetch", {"path": "notes6.txt"}),),
        prompt_tokens=OVER,
    )
    turns.append(_final())
    return turns


async def _stream(agent: Agent, task: str) -> list[AgentEvent]:
    return [event async for event in agent.stream(task)]


def _compactions(events: Sequence[AgentEvent]) -> list[AgentEvent]:
    return [event for event in events if event.type is AgentEventType.COMPACTION]


def _work(events: Sequence[AgentEvent]) -> list[AgentEvent]:
    """Compaction events that did something, or refused — not the informational ones."""

    return [
        event
        for event in _compactions(events)
        if not event.text.startswith(("unknown:", "within_budget:"))
    ]


def _sent(provider: MockProvider, index: int) -> list[Message]:
    request = provider.requests[index]
    messages: list[Message] = request["messages"]
    return messages


# --------------------------------------------------------------------------- L1


async def test_l1_truncates_a_stale_observation_to_head_and_tail(tmp_path: Path) -> None:
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool()])
    await agent.run("read notes")

    tool_message = next(m for m in _sent(agent.provider, 1) if m.role == "tool")
    assert tool_message.content.startswith(LONG_OBSERVATION[:HEAD_CHARS])
    assert tool_message.content.endswith(LONG_OBSERVATION[-TAIL_CHARS:])
    assert len(tool_message.content) < len(LONG_OBSERVATION) // 3
    assert "observation truncated by L1 compaction" in tool_message.content


async def test_the_replacement_names_the_call_that_produced_it(tmp_path: Path) -> None:
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool()])
    await agent.run("read notes")

    tool_message = next(m for m in _sent(agent.provider, 1) if m.role == "tool")
    # The pointer is a call the model can actually make: the tool exists under that
    # name, and the arguments shown are the ones that produced the dropped middle.
    assert "fetch" in tool_message.content
    assert "notes.txt" in tool_message.content
    assert tool_message.tool_call_id == "c1"


async def test_l1_leaves_short_observations_alone(tmp_path: Path) -> None:
    short = "a short observation"
    assert len(short) < MIN_OBSERVATION_CHARS
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool(observation=short)])
    result = await agent.run("read notes")

    # Compaction did run and chose not to touch it: the transcript it went on to
    # send still carries the observation whole.
    tool_message = next(m for m in result.messages if m.role == "tool")
    assert tool_message.content == short
    assert _work(result.events)


def test_l1_is_idempotent() -> None:
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", "", tool_calls=(tool_call("a", "fetch", {"path": "x"}),)),
        Message("tool", LONG_OBSERVATION, tool_call_id="a"),
    ]
    once = compact_l1(messages)
    twice = compact_l1(once.messages)
    assert once.chars_removed > 0
    assert twice.chars_removed == 0
    assert twice.messages == once.messages


# --------------------------------------------------------------------------- L2


def _transcript() -> list[Message]:
    """A system message, then six assistant turns each answered by a tool result."""

    messages: list[Message] = [Message("system", "sys"), Message("user", "the task")]
    for i in range(6):
        messages.append(Message("assistant", f"thought {i} " + "y" * 200))
        messages.append(Message("tool", f"observation {i} " + "z" * 200, tool_call_id=f"c{i}"))
    return messages


async def test_l2_keeps_the_last_turns_byte_identical() -> None:
    messages = _transcript()
    result = await compact_l2(messages, RecordingSummariser().summarise)

    assert result.turns_replaced > 0
    assert result.chars_removed > 0
    kept = KEEP_RECENT_TURNS * 2
    assert result.messages[-kept:] == messages[-kept:]


async def test_l2_replaces_the_middle_with_one_summary() -> None:
    result = await compact_l2(_transcript(), RecordingSummariser().summarise)

    summaries = [m for m in result.messages if m.content.startswith(_SUMMARY_PREFIX)]
    assert len(summaries) == 1
    assert "they read six files" in summaries[0].content
    assert summaries[0].role == "user"


async def test_l2_asks_a_model_to_summarise_the_middle_it_was_given() -> None:
    messages = _transcript()
    summariser = RecordingSummariser()
    await compact_l2(messages, summariser.summarise)

    assert len(summariser.shown) == 1
    asked = summariser.shown[0]
    # The summariser is shown the middle and nothing else: not the system prompt,
    # and not the recent turns that are kept verbatim anyway.
    assert asked[0].role == "assistant"
    assert all(not m.content.startswith(_SUMMARY_PREFIX) for m in asked)
    assert len(asked) < len(messages)


async def test_l2_keeps_the_system_message_first() -> None:
    messages = _transcript()
    result = await compact_l2(messages, RecordingSummariser().summarise)
    assert result.messages[0] == messages[0]


async def test_l2_refuses_to_spend_a_generation_on_nothing_to_say() -> None:
    summariser = RecordingSummariser()
    messages = [Message("system", "sys"), Message("user", "task"), Message("assistant", "done")]
    result = await compact_l2(messages, summariser.summarise)
    assert result.chars_removed == 0
    assert result.messages == messages
    assert summariser.shown == []


async def test_l2_refuses_a_summary_that_is_not_a_summary() -> None:
    result = await compact_l2(_transcript(), RecordingSummariser("   ").summarise)
    assert result.chars_removed == 0
    assert result.turns_replaced == 0


async def test_l2_does_not_summarise_its_own_summary() -> None:
    """A second L2 must not pay for a generation that re-summarises one summary."""

    summariser = RecordingSummariser()
    first = await compact_l2(_transcript(), summariser.summarise)
    assert first.chars_removed > 0
    second = await compact_l2(first.messages, summariser.summarise)
    assert len(summariser.shown) == 1
    assert second.messages == first.messages


# ---------------------------------------------------------------- the loop step


async def test_l1_runs_when_the_server_says_the_prompt_is_over_the_window(
    tmp_path: Path,
) -> None:
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool()])
    events = await _stream(agent, "read notes")

    (event,) = _work(events)
    assert event.text.startswith("l1:")
    assert "cut 1 observation(s)" in event.text
    assert "chars" in event.text
    assert f"{OVER} prompt tokens" in event.text
    assert f"window of {WINDOW}" in event.text
    assert "not built" not in event.text
    assert event.context is not None
    assert event.context.prompt_tokens == OVER


async def test_l2_runs_when_l1_has_nothing_left_to_cut(tmp_path: Path) -> None:
    summariser = RecordingSummariser()
    agent = _agent(
        _long_run(), tmp_path, tools=[FetchTool(observation="short")], summariser=summariser
    )
    events = await _stream(agent, "read notes")

    (event,) = _work(events)
    assert event.text.startswith("l1: nothing left to cut; l2:")
    assert "summarised 2 turn(s)" in event.text
    assert "removed" in event.text
    assert "chars" in event.text
    assert "summariser-model" in event.text
    assert summariser.shown, "L2 never asked a model to summarise"
    # The summariser is told what a summary is for, and is shown only the middle.
    assert summariser.shown[0][0].role == "system"
    assert "summarise" in summariser.shown[0][0].content.lower()


async def test_the_default_summariser_is_the_loop_own_provider(tmp_path: Path) -> None:
    """No second endpoint exists, so the honest default is the one endpoint there is."""

    agent = _agent(_long_run(), tmp_path, tools=[FetchTool(observation="short")])
    events = await _stream(agent, "read notes")

    # The summariser is the loop's own provider, so its model name is the loop's.
    l2 = [event for event in _work(events) if "l2:" in event.text]
    assert l2, "L2 never ran on the default summariser"
    assert "mock-model" in l2[0].text


async def test_compaction_is_not_a_per_iteration_tax(tmp_path: Path) -> None:
    """A within-budget iteration does no work, and a cut observation is never re-cut."""

    tool = FetchTool()
    agent = _agent(
        [_fetch_turn("c1", OVER), _fetch_turn("c2", WITHIN), _fetch_turn("c3", OVER), _final()],
        tmp_path,
        tools=[tool],
    )
    events = await _stream(agent, "read notes")

    texts = [event.text for event in _compactions(events)]
    assert len(texts) == 3
    assert texts[1].startswith("within_budget:")
    assert "nothing to compact" in texts[1]
    # One pass per over-budget measurement, not one per iteration: the within-budget
    # turn did no work at all.
    assert texts[0].startswith("l1: cut 1 observation(s)")
    assert texts[2].startswith("l1: cut 2 observation(s)")
    assert tool.calls == 3

    # Every long observation is truncated exactly once — no re-truncation of the
    # already-cut text, which would stack markers and shrink the transcript each pass.
    sent = _sent(agent.provider, -1)
    truncated = [m for m in sent if m.role == "tool" and "L1 compaction" in m.content]
    assert len(truncated) == 3
    for message in truncated:
        assert message.content.count("L1 compaction") == 1


async def test_an_unknown_context_size_is_reported_as_unknown(tmp_path: Path) -> None:
    for window, tokens, expected in (
        (None, OVER, "no context window"),
        (WINDOW, 0, "no prompt_tokens"),
    ):
        agent = _agent([_fetch_turn(prompt_tokens=tokens), _final()], tmp_path, window=window)
        events = await _stream(agent, "hi")
        (event,) = _compactions(events)
        assert event.text.startswith("unknown:"), (window, tokens, event.text)
        assert expected in event.text
        assert "nothing compacted" in event.text
        assert _work(events) == []


async def test_the_project_map_survives_l1(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nThe intro.\n", encoding="utf-8")
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool()])
    await agent.run("read notes")

    sent = _sent(agent.provider, 1)
    assert sent[0].role == "system"
    assert "# project map:" in sent[0].content
    assert "The intro." in sent[0].content
    # The task survived too: an oriented agent that forgot what it was asked is
    # no better than an unoriented one.
    assert sent[1].role == "user"
    assert sent[1].content == "read notes"


async def test_the_project_map_survives_l2(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\nThe intro.\n", encoding="utf-8")
    agent = _agent(
        _long_run(),
        tmp_path,
        tools=[FetchTool(observation="short")],
        summariser=RecordingSummariser(),
    )
    await agent.run("read notes")

    sent = _sent(agent.provider, -1)
    assert sent[0].role == "system"
    assert "# project map:" in sent[0].content
    assert "The intro." in sent[0].content
    assert sent[1].role == "user"
    assert sent[1].content == "read notes"
    # And the summary is in that same request: the map survived *with* the compaction.
    assert any(m.content.startswith(_SUMMARY_PREFIX) for m in sent)


async def test_the_summariser_usage_does_not_become_the_context_size(tmp_path: Path) -> None:
    """The summariser's prompt is a different request; its number is not how full we are."""

    agent = _agent(
        _long_run(),
        tmp_path,
        tools=[FetchTool(observation="short")],
        summariser=RecordingSummariser(),
    )
    result = await agent.run("read notes")

    (event,) = _work(result.events)
    assert event.context is not None
    assert event.context.prompt_tokens == OVER
    # The summariser's 500 prompt tokens still count against what the run cost.
    assert result.usage.prompt_tokens == 2 * OVER + 500


async def test_a_level_that_made_progress_lets_the_next_request_go_out(
    tmp_path: Path,
) -> None:
    """The loop does not invent a token count: it lets the server re-measure."""

    agent = _agent([_fetch_turn(OVER), _final(WITHIN)], tmp_path, tools=[FetchTool()])
    result = await agent.run("read notes")

    assert result.stop_reason is StopReason.COMPLETED
    assert len(agent.provider.requests) == 2


async def test_the_run_stops_when_compaction_cannot_fit_the_window(tmp_path: Path) -> None:
    summariser = RecordingSummariser()
    agent = _agent(
        [_fetch_turn(prompt_tokens=OVER)],
        tmp_path,
        tools=[FetchTool(observation="short")],
        summariser=summariser,
    )
    result = await agent.run("read notes")

    assert result.stop_reason is StopReason.CONTEXT_OVERFLOW
    assert result.ok is False
    assert result.error is not None
    assert "l1 and l2" in result.error
    assert f"{OVER} prompt tokens" in result.error
    assert f"window of {WINDOW}" in result.error
    assert "L3" in result.error
    assert "not built" in result.error
    (event,) = _work(result.events)
    assert event.text.startswith("l1: nothing left to cut; l2: no progress")
    # No generation is spent on an L2 that cannot have a middle.
    assert summariser.shown == []
    # And the request the server would have rejected was never sent.
    assert len(agent.provider.requests) == 1


# ------------------------------------------------------------------- the seams


def test_the_unbuilt_seam_is_exactly_l3() -> None:
    """Persistence shipped in m08c; only the L3 handoff may still be disclaimed."""
    assert len(UNBUILT_SEAMS) == 1
    joined = " ".join(UNBUILT_SEAMS).lower()
    assert "l3" in joined
    assert "persistence" not in joined
    assert "resume" not in joined
    # L1 and L2 may be *named* (they are L3's inputs) but neither may be reported
    # as missing. The subject of each seam is the text before "is not built".
    for seam in UNBUILT_SEAMS:
        subject = seam.lower().split("is not built")[0]
        assert "l1" not in subject
        assert "l2" not in subject
        assert "compaction" in subject


async def test_a_compaction_that_ran_never_reports_itself_as_unbuilt(tmp_path: Path) -> None:
    agent = _agent([_fetch_turn(), _final()], tmp_path, tools=[FetchTool()])
    events = await _stream(agent, "read notes")

    (event,) = _work(events)
    assert "not built" not in event.text
    assert "nothing compacted" not in event.text

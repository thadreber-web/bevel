"""L3: the handoff that replaces ``context_overflow`` with a continuation.

Nothing here asserts on the *wording* of what the model wrote. The brief embeds
the model's own account of its decisions verbatim, so grepping it for a phrase
would break on a rephrasing and prove nothing. What is pinned is provenance
(which list each part of the brief came from), shape (the rebuilt transcript has
the skeleton ``_seed()`` produces), budget accounting, the per-run cap, and
idempotency.

The trigger is still the server's own number, so the whole apparatus is
``ScriptedTurn(prompt_tokens=...)`` against ``Capabilities(max_model_len=...)``.
``OVER`` turns keep the ladder escalating; ``WITHIN`` turns let work accumulate
between handoffs, which is what makes a second one possible at all.

The two tests that matter most share a script and disagree only about the brief:
with a summariser that can write a short one the run completes, and with one
that writes a brief too big to fit the same run still stops with
``context_overflow``. That pair is the proof that the handoff, and not a lucky
fixture, is what continued the run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bevel.config.schema import MemoryConfig
from bevel.core.agent import (
    UNBUILT_SEAMS,
    Agent,
    AgentEvent,
    AgentEventType,
    Budgets,
    CompactionStatus,
    StopReason,
)
from bevel.core.context.compact import (
    _BRIEF_PREFIX,
    _SUMMARY_PREFIX,
    HANDOFF_KEEP_RECENT_TURNS,
    compact_l3,
)
from bevel.core.memory.store import MemoryStore, MemoryType
from bevel.core.messages import Message
from bevel.core.policy import allow_all_policy
from bevel.core.session.store import SessionStore, Todo, TodoStore
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.todo import TodoTool
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import AssistantTurn, Capabilities, GenOptions, Usage
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

OVER = 900
WITHIN = 50
WINDOW = 100
BRIEF_TOKENS = 500
LONG_SUMMARY = "summary " + "z" * 4000
SHORT_SUMMARY = "summary of the earlier turns"
BRIEF_ACCOUNT = "the parser was rewritten to stream; the tokenizer is still unresolved"


class RecordingSummariser:
    """Answers L2's request with a summary too big to help, and L3's with a short account.

    L2 must be exhausted for the ladder to reach L3, and the way it is exhausted
    is that its summary does not shrink the transcript. Branching on the system
    prompt is what a real second model does anyway: the two prompts ask for
    different things.
    """

    name = "summariser"
    model = "summariser-model"
    capabilities = Capabilities()

    def __init__(
        self,
        *,
        summary: str = LONG_SUMMARY,
        account: str = BRIEF_ACCOUNT,
        summaries: list[str] | None = None,
    ) -> None:
        self.summary = summary
        self.account = account
        # Answers L2 gets in sequence, then falls back to ``summary``. Lets a test
        # have one L2 pass actually land a summary and the next one come up empty,
        # which is the state a handoff has to clean up after.
        self._summaries = list(summaries or [])
        self.shown: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,  # noqa: ARG002
        options: GenOptions | None = None,  # noqa: ARG002
    ) -> AssistantTurn:
        self.shown.append(list(messages))
        first = messages[0].content if messages else ""
        if "handoff" in first.lower():
            text = self.account
        elif self._summaries:
            text = self._summaries.pop(0)
        else:
            text = self.summary
        return AssistantTurn(
            message=Message("assistant", text),
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=BRIEF_TOKENS, completion_tokens=20, total_tokens=BRIEF_TOKENS + 20
            ),
            model=self.model,
        )

    async def aclose(self) -> None:
        return None


class FetchTool:
    """A read tool with a short observation, so L1 never has anything to cut."""

    is_parallel_safe = True

    def __init__(self, tier: RiskTier = RiskTier.READ) -> None:
        self.risk_tier = tier
        self.spec = ToolSpec(
            name="fetch",
            description="Fetch a path.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        self.calls = 0

    async def execute(self, arguments: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        del ctx
        self.calls += 1
        return ToolResult(ok=True, content=f"read {arguments.get('path')}")


class WriteTool(FetchTool):
    """The same tool declared as a workspace writer, so a handoff has a file to name.

    ``path`` is deliberately *not* required by the schema: a call that satisfies
    the schema while carrying no usable path is the case the brief has to survive,
    and making it a validation error would hide it behind the repair layer.
    """

    is_parallel_safe = False

    def __init__(self) -> None:
        super().__init__(tier=RiskTier.WRITE_WORKSPACE)
        self.spec = ToolSpec(
            name="write",
            description="Write a path.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        )

    async def execute(self, arguments: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        del ctx
        self.calls += 1
        path = arguments.get("path")
        if not isinstance(path, str):
            return ToolResult(ok=True, content="wrote an unnamed file")
        return ToolResult(ok=True, content=f"wrote {path}")


def _agent(
    provider: MockProvider,
    tmp_path: Path,
    *,
    tools: list[Any] | None = None,
    summariser: Any = None,
    budgets: Budgets | None = None,
    **kwargs: Any,
) -> Agent:
    budgets = budgets or Budgets(max_iterations=40, max_tool_calls=200)
    provider.capabilities = Capabilities(
        max_model_len=WINDOW, supports_tools=True, supports_streaming=True
    )
    return Agent(
        provider,
        tmp_path,
        tools=tools or [],
        budgets=budgets,
        policy=allow_all_policy,
        summariser=summariser,
        **kwargs,
    )


def _work(i: int, prompt_tokens: int = WITHIN) -> ScriptedTurn:
    """One prose-plus-tool turn. The prose is what L2 would have summarised."""
    return ScriptedTurn(
        content=f"thinking about file {i} " + "y" * 300,
        tool_calls=(tool_call(f"c{i}", "fetch", {"path": f"notes{i}.txt"}),),
        prompt_tokens=prompt_tokens,
    )


def _over_work(i: int) -> ScriptedTurn:
    return _work(i, OVER)


def _final(prompt_tokens: int = WITHIN) -> ScriptedTurn:
    return ScriptedTurn(content="done", prompt_tokens=prompt_tokens)


def _run_script(*, handoffs: int, lead: Sequence[ScriptedTurn] = ()) -> list[ScriptedTurn]:
    """A script that forces ``handoffs`` handoffs and then finishes.

    ``lead`` turns run first, so a test can have a write or a todo call happen
    before the handoff that has to account for it.

    Each handoff is triggered by one ``OVER`` turn, taken when the transcript
    holds ``HANDOFF_KEEP_RECENT_TURNS + 2`` assistant turns. That is the point
    the ladder actually reaches L3: L1 has no long observation to cut, L2 has
    nothing left to fold into a summary because it keeps that many turns itself,
    and L3 -- which keeps fewer -- still has a region above its own tail.
    """
    turns: list[ScriptedTurn] = list(lead)
    n = len(turns)
    for _ in range(handoffs):
        turns.extend(_work(n + k) for k in range(HANDOFF_KEEP_RECENT_TURNS + 1))
        n += HANDOFF_KEEP_RECENT_TURNS + 1
        turns.append(_over_work(n))
        n += 1
    turns.append(_final())
    return turns


def _sent(provider: MockProvider, index: int) -> list[Message]:
    request: dict[str, object] = provider.requests[index]
    return request["messages"]  # type: ignore[return-value]


def _brief_request(summariser: RecordingSummariser, which: int = 0) -> list[Message]:
    """The request that asked the model for its account of the handoff.

    It goes to the summariser, not the provider — the same seam L2's summariser
    uses — so it is found in the summariser's log, and the only thing that tells
    the two kinds of request apart is the system prompt.
    """
    hits = [
        shown
        for shown in summariser.shown
        if shown and shown[0].role == "system" and "handoff" in shown[0].content.lower()
    ]
    assert hits, "no handoff brief was ever requested"
    return hits[which]


def _rebuilt_index(provider: MockProvider, which: int = 0) -> int:
    """The provider request made *after* a handoff: the one carrying the brief."""
    indices = [
        i
        for i in range(len(provider.requests))
        if any(message.content.startswith(_BRIEF_PREFIX) for message in _sent(provider, i))
    ]
    assert indices, "no rebuilt transcript was ever sent"
    return indices[which]


def _rebuilt(provider: MockProvider, which: int = 0) -> list[Message]:
    return _sent(provider, _rebuilt_index(provider, which))


def _compactions(result: Any) -> list[AgentEvent]:
    return [event for event in result.events if event.type is AgentEventType.COMPACTION]


def _handoffs(result: Any) -> list[AgentEvent]:
    """The announcements of handoffs that actually happened.

    Matched on the words only a completed handoff uses, so a level that refused
    or found nothing to do is not counted as one.
    """
    return [event for event in _compactions(result) if "handoff replaced" in event.text.lower()]


# ---------------------------------------------------------------------------
# The test that matters most, and its control.
# ---------------------------------------------------------------------------


async def test_a_run_that_would_have_overflowed_now_completes(tmp_path: Path) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser())

    result = await agent.run("fix the flaky parser")

    assert result.stop_reason is StopReason.COMPLETED, result.error
    assert result.final_text == "done"
    # L1 and L2 were both reached and both came up empty first, so this is the
    # transcript that used to die rather than one that never needed a handoff.
    texts = [event.text for event in _compactions(result)]
    assert any("l1" in text and "nothing" in text for text in texts), texts
    assert any("l2" in text and "no progress" in text for text in texts), texts
    assert len(_handoffs(result)) == 1, texts


async def test_the_same_run_still_overflows_when_the_brief_will_not_fit(tmp_path: Path) -> None:
    """The control. Identical script, useless brief, honest stop.

    Without this the test above could pass because the fixture was small rather
    than because the handoff worked.
    """
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=RecordingSummariser(account="brief " + "b" * 200_000),
    )

    result = await agent.run("fix the flaky parser")

    assert result.stop_reason is StopReason.CONTEXT_OVERFLOW
    assert result.error is not None
    assert "handoff" in result.error.lower()


# ---------------------------------------------------------------------------
# Shape of the rebuilt transcript.
# ---------------------------------------------------------------------------


async def test_the_rebuilt_transcript_has_the_opening_shape(tmp_path: Path) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser())
    result = await agent.run("fix the flaky parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    rebuilt = _rebuilt(provider)

    assert rebuilt[0].role == "system"
    assert "# project map:" in rebuilt[0].content, "the handoff reused a stale project map"
    assert rebuilt[1].role == "user"
    brief = rebuilt[1].content
    assert "handoff" in brief.lower()
    for heading in ("goal", "decisions", "files touched", "open questions", "todos"):
        assert heading in brief.lower(), heading
    assert "fix the flaky parser" in brief, "the goal did not come from the task verbatim"
    # The model's account is embedded verbatim under a heading naming it as the
    # model's own. Provenance is structural, not a phrase to grep for.
    assert BRIEF_ACCOUNT in brief


async def test_the_kept_tail_is_the_last_turns_verbatim(tmp_path: Path) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser())
    result = await agent.run("fix the flaky parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    at = _rebuilt_index(provider)
    rebuilt = _sent(provider, at)
    final_assistants = [m for m in result.messages if m.role == "assistant"]

    kept = [m for m in rebuilt if m.role == "assistant"]
    # Exactly the handoff's tail, carried through unchanged: they are the same
    # messages the rest of the run then continued from, not a rewrite of them.
    assert len(kept) == HANDOFF_KEEP_RECENT_TURNS
    assert kept == final_assistants[:HANDOFF_KEEP_RECENT_TURNS], (
        "the kept tail must be the same messages, byte-identical"
    )
    # ...and they are the *last* turns of the old transcript, not the first: what
    # L2 could not fold away is gone.
    assert not any("thinking about file 0" in m.content for m in rebuilt)
    # L3 keeps fewer turns than L2, which is what lets it shrink the transcript
    # L2 gave up on. Were the two levels to keep the same tail, escalation
    # could never reach this branch.
    assert len(kept) < len(final_assistants)
    # Nothing may be orphaned: every call asked for still has its result beside it.
    asked = {call.id for m in rebuilt if m.role == "assistant" for call in m.tool_calls}
    answered = {m.tool_call_id for m in rebuilt if m.role == "tool"}
    assert asked == answered


async def test_the_recall_block_lands_at_index_one_again(tmp_path: Path) -> None:
    matched = "the parser broke on streaming tokens"
    memory = MemoryStore(tmp_path / "memories.db", config=MemoryConfig(enabled=True))
    memory.add(content=matched, type=MemoryType.FACT, scope="global")

    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=RecordingSummariser(),
        memory=memory,
    )
    result = await agent.run("why does the parser break on streaming tokens")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    rebuilt = _rebuilt(provider)
    assert rebuilt[1].role == "user"
    assert matched in rebuilt[1].content, "the handoff did not re-run recall"
    assert "handoff" in (rebuilt[2].content or "").lower()
    # The query is the current state, not the original task: the memory that
    # shares its words with what the agent was last doing is the one that landed.


async def test_the_brief_replaces_the_l2_summary_it_supersedes(tmp_path: Path) -> None:
    # The first over-budget measurement lands where L2 can still help, so a real
    # summary goes in. The second lands where it cannot, and the handoff that
    # follows has to drop the summary it supersedes rather than ship both.
    turns = [_work(0), _work(1), _work(2), _work(3), _over_work(4), _over_work(5), _final()]
    provider = MockProvider(turns=turns)
    summariser = RecordingSummariser(summaries=[SHORT_SUMMARY, LONG_SUMMARY])
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=summariser)
    result = await agent.run("fix the flaky parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    rebuilt = _rebuilt(provider)
    joined = "\n".join(m.content for m in rebuilt)
    assert _SUMMARY_PREFIX not in joined, (
        "a superseded L2 summary survived the handoff and is still on the wire"
    )
    assert SHORT_SUMMARY not in joined
    before = _sent(provider, _rebuilt_index(provider) - 1)
    assert any(m.content.startswith(_SUMMARY_PREFIX) for m in before), (
        "the control is hollow: L2 never produced a summary for the handoff to replace"
    )


# ---------------------------------------------------------------------------
# Provenance: which list each part of the brief came from.
# ---------------------------------------------------------------------------


async def test_files_touched_are_only_files_a_tool_actually_mutated(tmp_path: Path) -> None:
    summariser = RecordingSummariser()
    provider = MockProvider(
        turns=_run_script(
            handoffs=1,
            lead=[
                ScriptedTurn(
                    content="writing " + "y" * 300,
                    tool_calls=(tool_call("w1", "write", {"path": "src/a.py"}),),
                    prompt_tokens=WITHIN,
                )
            ],
        )
    )
    agent = _agent(provider, tmp_path, tools=[FetchTool(), WriteTool()], summariser=summariser)
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    files = _section(_brief_request(summariser)[1].content, "files touched")
    assert "src/a.py" in files
    assert "notes1.txt" not in files, "a file that was only read is listed as touched"


async def test_a_mutated_call_with_unusable_arguments_contributes_no_path(tmp_path: Path) -> None:
    """A write whose arguments name no path contributes nothing — never a guess."""
    summariser = RecordingSummariser()
    provider = MockProvider(
        turns=_run_script(
            handoffs=1,
            lead=[
                # A write the tool refused for want of a path: it never mutated,
                # so it must contribute nothing rather than a guessed name.
                ScriptedTurn(
                    content="writing " + "y" * 300,
                    tool_calls=(tool_call("w1", "write", {}),),
                    prompt_tokens=WITHIN,
                ),
                ScriptedTurn(
                    content="writing again " + "y" * 300,
                    tool_calls=(tool_call("w2", "write", {"path": "src/b.py"}),),
                    prompt_tokens=WITHIN,
                ),
            ],
        )
    )
    agent = _agent(provider, tmp_path, tools=[WriteTool()], summariser=summariser)
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    files = _section(_brief_request(summariser)[1].content, "files touched")
    assert "src/b.py" in files
    assert "unknown" not in files.lower()
    assert "src/a.py" not in files


async def test_the_todos_come_from_the_store(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    store, todos = SessionStore(db), TodoStore(db)
    summariser = RecordingSummariser()
    provider = MockProvider(
        turns=_run_script(
            handoffs=1,
            lead=[
                ScriptedTurn(
                    content="planning " + "y" * 300,
                    tool_calls=(
                        tool_call(
                            "t1",
                            "todo",
                            {"todos": [{"content": "wire the lexer", "status": "in_progress"}]},
                        ),
                    ),
                    prompt_tokens=WITHIN,
                )
            ],
        )
    )
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool(), TodoTool()],
        summariser=summariser,
        store=store,
        todos=todos,
    )
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    section = _section(_brief_request(summariser)[1].content, "todos")
    assert "wire the lexer" in section
    assert "in_progress" in section
    store.close()
    todos.close()


async def test_a_run_with_no_todos_says_so(tmp_path: Path) -> None:
    summariser = RecordingSummariser()
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=summariser)
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    section = _section(_brief_request(summariser)[1].content, "todos")
    assert "no todo list" in section.lower()


def _section(brief: str, heading: str) -> str:
    """The text under ``## heading`` up to the next heading."""
    _, _, rest = brief.partition(f"## {heading}")
    assert rest, f"the brief has no {heading!r} section"
    return rest.split("\n## ", 1)[0]


# ---------------------------------------------------------------------------
# Budget accounting.
# ---------------------------------------------------------------------------


async def test_the_brief_request_is_billed_but_does_not_move_the_context_reading(
    tmp_path: Path,
) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    summariser = RecordingSummariser()
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=summariser)

    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    # The brief request really happened and really cost something...
    assert result.usage.prompt_tokens >= BRIEF_TOKENS * len(summariser.shown)
    # ...but its prompt is not the next thing the model saw, so no compaction
    # event may report the summariser's size as the window reading. Every
    # over-budget measurement here was taken on a request whose prompt_tokens
    # the loop had just read from the server, so all of them read OVER.
    readings = [
        event.context.prompt_tokens
        for event in _compactions(result)
        if event.context is not None and event.context.status is not CompactionStatus.WITHIN_BUDGET
    ]
    assert readings
    assert all(reading == OVER for reading in readings), readings


async def test_a_handoff_costs_an_iteration(tmp_path: Path) -> None:
    # Two turns of work and then the measurement that trips the ladder: three
    # assistant turns is the smallest transcript L2 has given up on and L3 has
    # not, so the handoff happens right there and nothing is streamed after it.
    provider = MockProvider(turns=[_work(0), _work(1), _over_work(2)])
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=RecordingSummariser(),
        budgets=Budgets(max_iterations=4, max_tool_calls=200),
    )

    result = await agent.run("fix the parser")

    # Three completions, and the handoff consumed the fourth of the four allowed
    # iterations. Free work would have got the model another completion instead.
    assert len(provider.requests) == 3
    assert len(_handoffs(result)) == 1
    assert result.iterations == 4
    assert result.stop_reason is StopReason.TURN_BUDGET


# ---------------------------------------------------------------------------
# The cap.
# ---------------------------------------------------------------------------


async def test_the_cap_stops_the_run_with_a_named_reason(tmp_path: Path) -> None:
    turns = _run_script(handoffs=4)[:-1]  # never finishes: nothing but work and overflow
    provider = MockProvider(turns=turns)
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=RecordingSummariser(),
        budgets=Budgets(max_iterations=200, max_tool_calls=500),
    )

    result = await agent.run("fix the parser")

    assert result.stop_reason is StopReason.CONTEXT_OVERFLOW
    assert result.error is not None
    assert "cap" in result.error.lower()
    handoffs = _handoffs(result)
    assert len(handoffs) == 2, [event.text for event in handoffs]


# ---------------------------------------------------------------------------
# Idempotency, pinned the way L1 and L2 are.
# ---------------------------------------------------------------------------


def _transcript(turns: int) -> list[Message]:
    out: list[Message] = [Message("system", "system prompt\n\n# project map: /w")]
    for i in range(turns):
        out.append(
            Message(
                "assistant",
                f"turn {i} " + "y" * 400,
                tool_calls=(tool_call(f"c{i}", "fetch", {"path": "x"}),),
            )
        )
        out.append(Message("tool", f"observation {i}", tool_call_id=f"c{i}"))
    return out


def test_l3_is_idempotent() -> None:
    messages = _transcript(6)
    head = [Message.system("system prompt\n\n# project map: /w")]
    once = compact_l3(messages, head=head, brief="## goal\n\nthe task")
    assert once.changed
    assert once.chars_removed > 0
    assert once.turns_replaced == 6 - HANDOFF_KEEP_RECENT_TURNS

    twice = compact_l3(once.messages, head=head, brief="## goal\n\nthe task")
    assert not twice.changed
    assert twice.messages == once.messages
    assert twice.reason == "the transcript is already a handoff with nothing new to fold in"


def test_l3_refuses_a_brief_that_does_not_shrink_the_transcript() -> None:
    messages = _transcript(6)
    head = [Message.system("system prompt\n\n# project map: /w")]
    result = compact_l3(messages, head=head, brief="x" * 200_000)
    assert not result.changed
    assert result.messages == messages
    assert result.turns_replaced == 0
    assert "fit" in result.reason.lower()


def test_l3_has_nothing_to_do_when_there_are_no_turns_to_fold() -> None:
    head = [Message.system("system prompt")]
    result = compact_l3(head, head=head, brief="## goal\n\nthe task")
    assert not result.changed
    assert result.reason == "there are no turns to hand off"


# ---------------------------------------------------------------------------
# Degradation, never fabrication.
# ---------------------------------------------------------------------------


async def test_a_failing_brief_request_falls_back_to_overflow(tmp_path: Path) -> None:
    class Broken:
        name = "broken"
        model = "broken-model"
        capabilities = Capabilities()

        async def complete(
            self,
            messages: Sequence[Message],  # noqa: ARG002
            tools: Sequence[ToolSpec] | None = None,  # noqa: ARG002
            options: GenOptions | None = None,  # noqa: ARG002
        ) -> AssistantTurn:
            raise RuntimeError("the endpoint is down")

        async def aclose(self) -> None:
            return None

    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=Broken())
    result = await agent.run("fix the parser")

    assert result.stop_reason is StopReason.CONTEXT_OVERFLOW
    assert result.error is not None
    assert "the endpoint is down" in result.error


async def test_an_empty_brief_falls_back_to_overflow(tmp_path: Path) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser(account="   ")
    )
    result = await agent.run("fix the parser")

    assert result.stop_reason is StopReason.CONTEXT_OVERFLOW
    assert result.error is not None
    assert "empty" in result.error.lower()


async def test_a_broken_memory_store_does_not_cost_the_task(tmp_path: Path) -> None:
    class BrokenMemory:
        config = MemoryConfig(enabled=True)
        embedder = None

        def candidates(
            self,
            *args: object,  # noqa: ARG002
            **kwargs: object,  # noqa: ARG002
        ) -> list[object]:
            raise RuntimeError("the memory index is corrupt")

    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=RecordingSummariser(),
        memory=BrokenMemory(),  # type: ignore[arg-type]
    )
    result = await agent.run("fix the parser")

    assert result.stop_reason is StopReason.COMPLETED, result.error
    rebuilt = _rebuilt(provider)
    assert rebuilt[0].role == "system"
    assert "handoff" in (rebuilt[1].content or "").lower()


async def test_a_broken_todo_store_says_the_list_could_not_be_read(tmp_path: Path) -> None:
    class BrokenTodos:
        session_id: str | None = None

        def list_todos(self) -> tuple[Todo, ...]:
            raise RuntimeError("the todo table is gone")

    summariser = RecordingSummariser()
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider,
        tmp_path,
        tools=[FetchTool()],
        summariser=summariser,
        todos=BrokenTodos(),  # type: ignore[arg-type]
    )
    result = await agent.run("fix the parser")

    assert result.stop_reason is StopReason.COMPLETED, result.error
    section = _section(_brief_request(summariser)[1].content, "todos")
    assert "could not be read" in section.lower()


# ---------------------------------------------------------------------------
# Nothing is invented, and nothing is silently dropped.
# ---------------------------------------------------------------------------


async def test_the_brief_is_not_persisted_as_a_turn(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    store = SessionStore(db)
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(
        provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser(), store=store
    )
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error
    assert result.session_id is not None

    rows = store.turns(result.session_id)
    stored = "\n".join(row.content for row in rows)
    assert BRIEF_ACCOUNT not in stored
    assert "## files touched" not in stored
    assert [row.role for row in rows][:2] == ["user", "assistant"]
    store.close()


async def test_the_handoff_is_announced_with_numbers(tmp_path: Path) -> None:
    provider = MockProvider(turns=_run_script(handoffs=1))
    agent = _agent(provider, tmp_path, tools=[FetchTool()], summariser=RecordingSummariser())
    result = await agent.run("fix the parser")
    assert result.stop_reason is StopReason.COMPLETED, result.error

    (event,) = _handoffs(result)
    assert "l3" in event.text.lower()
    assert "removed" in event.text.lower()
    assert "not built" not in event.text.lower()
    assert event.context is not None
    assert event.context.prompt_tokens == OVER


async def test_the_unbuilt_seam_no_longer_names_l3() -> None:
    assert not any("l3" in seam.lower() for seam in UNBUILT_SEAMS), UNBUILT_SEAMS

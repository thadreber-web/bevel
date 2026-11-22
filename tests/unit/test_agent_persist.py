"""The loop writing through to the session store as it goes.

The promise m08c makes is that a turn which completed is in the file. That is
only testable by running the loop against a real SQLite file and reading the
rows back with a second connection, which is what these do. A store the loop
*could* write to is not the same thing as a store the loop writes to.

Resume is tested here too, at the seam that matters: the messages a previous
run left behind are what the next request is built on.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from bevel.core.agent import (
    Agent,
    AgentEvent,
    AgentEventType,
    Budgets,
    Decision,
    PolicyRequest,
    PolicyVerdict,
    StopReason,
)
from bevel.core.context.compact import HEAD_CHARS, MIN_OBSERVATION_CHARS, compact_l1
from bevel.core.messages import Message, ToolCall
from bevel.core.session.resume import rebuild_transcript
from bevel.core.session.store import SessionStore
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import Capabilities
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

OBSERVATION = "1|one\n2|two"


class FetchTool:
    """A read tool with a fixed observation, so a call has something to record."""

    is_parallel_safe = True

    def __init__(self, observation: str = OBSERVATION) -> None:
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


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "bevel" / "sessions.db")


def _stored_workspace(path: Path) -> str:
    """What the store records for a workspace: absolute and resolved."""
    return str(path.expanduser().resolve(strict=False))


def _agent(
    turns: list[ScriptedTurn],
    tmp_path: Path,
    *,
    tools: list[object] | None = None,
    **kwargs: object,
) -> Agent:
    provider = MockProvider(turns=turns)
    provider.capabilities = Capabilities(supports_tools=True, supports_streaming=True)
    return Agent(provider, tmp_path, tools=tools or [], **kwargs)  # type: ignore[arg-type]


def _ask(call_id: str = "c1", name: str = "fetch", **arguments: object) -> ScriptedTurn:
    return ScriptedTurn(content="", tool_calls=(tool_call(call_id, name, arguments),))


def _sent(provider: MockProvider, index: int) -> list[Message]:
    request = provider.requests[index]
    messages: list[Message] = request["messages"]  # type: ignore[assignment]
    return messages


def _rows(store: SessionStore, session_id: str) -> list[Message]:
    """The transcript as a second connection sees it."""
    reopened = SessionStore(store.db_path)
    return [
        Message(role=turn.role, content=turn.content, tool_call_id=turn.tool_call_id)  # type: ignore[arg-type]
        for turn in reopened.turns(session_id)
    ]


async def test_a_completed_run_is_fully_in_the_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _agent(
        [ScriptedTurn(content="hello there", prompt_tokens=11, completion_tokens=4)],
        tmp_path,
        store=store,
    )
    result = await agent.run("greet me")

    assert result.session_id == agent.session_id
    assert [m.role for m in _rows(store, result.session_id or "")] == ["user", "assistant"]
    assert [m.content for m in _rows(store, result.session_id or "")] == [
        "greet me",
        "hello there",
    ]

    (listed,) = [
        s for s in SessionStore(store.db_path).list_sessions() if s.id == result.session_id
    ]
    assert (listed.prompt_tokens, listed.completion_tokens, listed.total_tokens) == (11, 4, 15)
    assert listed.turns == 2
    assert listed.workspace == _stored_workspace(tmp_path)
    assert listed.provider == "mock"
    assert listed.model == "mock-model"


async def test_the_system_message_and_project_map_are_not_stored(tmp_path: Path) -> None:
    """The map is rebuilt each run; storing a stale one would resurrect it on resume."""
    store = _store(tmp_path)
    result = await _agent([ScriptedTurn(content="ok")], tmp_path, store=store).run("task")
    assert [m.role for m in _rows(store, result.session_id or "")] == ["user", "assistant"]


async def test_a_tool_call_is_stored_with_its_arguments_outcome_and_observation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    agent = _agent(
        [_ask(path="notes.txt"), ScriptedTurn(content="done")],
        tmp_path,
        tools=[FetchTool()],
        store=store,
    )
    result = await agent.run("read notes")

    turns = SessionStore(store.db_path).turns(result.session_id or "")
    assert [turn.role for turn in turns] == ["user", "assistant", "tool", "assistant"]
    (call,) = next(turn for turn in turns if turn.tool_calls).tool_calls
    assert (call.call_id, call.name, call.outcome) == ("c1", "fetch", "ok")
    assert call.arguments == '{"path": "notes.txt"}'
    assert call.observation == OBSERVATION


async def test_a_refused_call_is_stored_with_its_reason(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def deny(request: PolicyRequest) -> PolicyVerdict:
        del request
        return PolicyVerdict(Decision.DENY, "not in this workspace")

    agent = _agent(
        [_ask(path="x"), ScriptedTurn(content="understood")],
        tmp_path,
        tools=[FetchTool()],
        store=store,
        policy=deny,
    )
    result = await agent.run("read")

    turns = SessionStore(store.db_path).turns(result.session_id or "")
    (call,) = next(turn for turn in turns if turn.tool_calls).tool_calls
    assert call.outcome == "rejected"
    assert call.reason == "not in this workspace"


async def test_a_first_time_right_call_records_no_fixes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _agent(
        [_ask(path="notes.txt"), ScriptedTurn(content="done")],
        tmp_path,
        tools=[FetchTool()],
        store=store,
    )
    result = await agent.run("read notes")

    turns = SessionStore(store.db_path).turns(result.session_id or "")
    (call,) = next(turn for turn in turns if turn.tool_calls).tool_calls
    assert call.fixes == ()


async def test_every_iteration_gets_its_own_usage_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _agent(
        [
            ScriptedTurn(
                content="",
                tool_calls=(tool_call("c1", "fetch", {"path": "n"}),),
                prompt_tokens=50,
                completion_tokens=5,
            ),
            ScriptedTurn(content="done", prompt_tokens=80, completion_tokens=7),
        ],
        tmp_path,
        tools=[FetchTool()],
        store=store,
    )
    result = await agent.run("read notes")

    rows = SessionStore(store.db_path).usage_rows(result.session_id or "")
    assert [(r.iteration, r.prompt_tokens, r.completion_tokens) for r in rows] == [
        (1, 50, 5),
        (2, 80, 7),
    ]


async def test_a_run_with_no_store_persists_nothing_but_still_runs(tmp_path: Path) -> None:
    """Attaching a store is optional; the loop must not require one."""
    result = await _agent([ScriptedTurn(content="fine")], tmp_path).run("are you there")
    assert result.ok
    assert result.session_id is None
    assert not (tmp_path / "bevel").exists()


async def test_a_resumed_run_builds_its_first_request_on_the_stored_history(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = _agent([ScriptedTurn(content="the secret word is zephyr")], tmp_path, store=store)
    before = await first.run("remember the word")

    resumed = rebuild_transcript(SessionStore(store.db_path), before.session_id or "")
    assert resumed is not None
    provider = MockProvider([ScriptedTurn(content="zephyr")])
    second = Agent(provider, tmp_path, tools=[], store=SessionStore(store.db_path), resume=resumed)
    after = await second.run("what was the word")

    sent = _sent(provider, 0)
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[1].content == "remember the word"
    assert sent[2].content == "the secret word is zephyr"
    assert sent[3].content == "what was the word"
    assert after.session_id == before.session_id


async def test_a_resumed_run_appends_rather_than_rewriting(tmp_path: Path) -> None:
    store = _store(tmp_path)
    before = await _agent([ScriptedTurn(content="one")], tmp_path, store=store).run("first")

    resumed = rebuild_transcript(SessionStore(store.db_path), before.session_id or "")
    assert resumed is not None
    second = _agent([ScriptedTurn(content="two")], tmp_path, store=store, resume=resumed)
    after = await second.run("second")

    assert [m.content for m in _rows(store, after.session_id or "")] == [
        "first",
        "one",
        "second",
        "two",
    ]
    assert len(store.list_sessions()) == 1


async def test_a_resumed_run_drops_an_unusable_tail_rather_than_sending_it(
    tmp_path: Path,
) -> None:
    """A crash mid-batch leaves a call with no result; resume must not put it on the wire."""
    store = _store(tmp_path)
    before = await _agent([ScriptedTurn(content="one")], tmp_path, store=store).run("first")
    store.append_turn(
        before.session_id or "",
        Message("assistant", tool_calls=(ToolCall("c-dangling", "read", "{}"),)),
        iteration=9,
    )

    resumed = rebuild_transcript(SessionStore(store.db_path), before.session_id or "")
    assert resumed is not None
    assert resumed.dropped == 1
    assert [m.content for m in resumed.messages] == ["first", "one"]

    provider = MockProvider([ScriptedTurn(content="two")])
    await Agent(provider, tmp_path, tools=[], store=store, resume=resumed).run("second")
    assert all(m.role != "tool" for m in _sent(provider, 0))


async def test_a_run_that_stops_early_still_leaves_its_completed_turn(tmp_path: Path) -> None:
    """Crash-safety: the assistant turn and its calls are written before the next request."""
    store = _store(tmp_path)
    agent = _agent(
        [_ask(path="notes.txt"), ScriptedTurn(content="never reached")],
        tmp_path,
        tools=[FetchTool()],
        store=store,
        budgets=Budgets(max_iterations=1),
    )
    events = [event async for event in agent.stream("read notes")]

    assert any(e.type is AgentEventType.COMPLETED for e in events)
    turns = SessionStore(store.db_path).turns(agent.session_id or "")
    assert [turn.role for turn in turns] == ["user", "assistant", "tool"]
    assert turns[1].tool_calls[0].call_id == "c1"
    assert turns[1].tool_calls[0].outcome == "ok"


async def test_a_cancelled_run_leaves_what_completed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    agent = _agent(
        [_ask(path="notes.txt"), ScriptedTurn(content="late")],
        tmp_path,
        tools=[FetchTool()],
        store=store,
    )
    events: list[AgentEvent] = []
    async for event in agent.stream("read notes"):
        events.append(event)
        if event.type is AgentEventType.TOOL_COMPLETE:
            agent.cancel("user pressed ctrl+c")

    completed = [e for e in events if e.type is AgentEventType.COMPLETED]
    assert completed[0].text == StopReason.CANCELLED.value
    turns = SessionStore(store.db_path).turns(agent.session_id or "")
    assert [turn.role for turn in turns] == ["user", "assistant", "tool"]


def test_l1_points_at_the_stored_copy_of_a_dropped_observation() -> None:
    """L1's replacement must name the saved text, not claim it was thrown away."""
    long = "x" * (MIN_OBSERVATION_CHARS + 500)
    messages = [
        Message("system", "sys"),
        Message("user", "task"),
        Message("assistant", "", tool_calls=(ToolCall("a", "read", '{"path": "big.txt"}'),)),
        Message("tool", long, tool_call_id="a"),
    ]
    stored = compact_l1(messages, session_id="abc123def456")
    text = stored.messages[3].content
    assert text.startswith(long[:HEAD_CHARS])
    assert "abc123def456" in text
    assert "was not saved" not in text
    assert "read(" in text

    unattached = compact_l1(messages)
    assert "was not saved" in unattached.messages[3].content
    assert "abc123def456" not in unattached.messages[3].content


async def test_l1_in_the_loop_points_at_the_live_session(tmp_path: Path) -> None:
    long = "OBSERVATION\n" + "\n".join(f"line {i} {'x' * 40}" for i in range(120))
    store = _store(tmp_path)
    # The trigger is the server's own prompt_tokens, so the turn has to report one.
    provider = MockProvider(
        [
            ScriptedTurn(
                content="",
                tool_calls=(tool_call("c1", "fetch", {"path": "big.txt"}),),
                prompt_tokens=900,
            ),
            ScriptedTurn(content="done", prompt_tokens=900),
        ]
    )
    provider.capabilities = Capabilities(
        max_model_len=100, supports_tools=True, supports_streaming=True
    )
    agent = Agent(provider, tmp_path, tools=[FetchTool(observation=long)], store=store)
    result = await agent.run("read")

    tool_message = next(m for m in result.messages if m.role == "tool")
    assert "observation truncated by L1 compaction" in tool_message.content
    assert result.session_id is not None
    assert result.session_id in tool_message.content
    assert "was not saved" not in tool_message.content
    # The full text is still in the store even though the wire copy was cut.
    turns = SessionStore(store.db_path).turns(result.session_id)
    assert any(len(turn.content) == len(long) for turn in turns)

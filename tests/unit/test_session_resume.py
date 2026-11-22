"""Rebuilding a transcript from the store so a closed session can continue.

The property that matters is the one a real server enforces with a 400: an
assistant message that asked for tools must be followed by a result for every
call it named. A crash mid-batch therefore leaves a turn that cannot be sent,
and resume has to notice and drop it rather than hand the loop a transcript the
endpoint will reject.

These tests write rows through the real store and read messages back, so they
fail if resume invents history, reorders it, or keeps a dangling call.
"""

from __future__ import annotations

from pathlib import Path

from bevel.core.agent import ToolExecution, ToolOutcome
from bevel.core.messages import Message, ToolCall
from bevel.core.session.resume import ResumedSession, rebuild_transcript
from bevel.core.session.store import SessionStore
from bevel.core.tools.base import ToolResult


def _store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def _execution(
    call_id: str,
    name: str = "read",
    arguments: str = '{"path": "a.py"}',
    observation: str = "1|one",
    outcome: ToolOutcome = ToolOutcome.OK,
) -> ToolExecution:
    return ToolExecution(
        call=ToolCall(id=call_id, name=name, arguments=arguments),
        tool=None,
        outcome=outcome,
        result=ToolResult(ok=outcome is ToolOutcome.OK, content=observation),
    )


def _ask(store: SessionStore, session_id: str, iteration: int, *calls: ToolCall) -> int:
    return store.append_turn(
        session_id, Message("assistant", tool_calls=calls), iteration=iteration
    )


def _answer(store: SessionStore, session_id: str, turn: int, *executions: ToolExecution) -> None:
    for execution in executions:
        store.append_tool_call(session_id, turn, execution)
        store.append_turn(
            session_id,
            Message.tool_result(
                execution.call.id, execution.result.content if execution.result else ""
            ),
            iteration=turn,
        )


def test_a_simple_session_rebuilds_in_order(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(info.id, Message.user("hello"), iteration=1)
    store.append_turn(info.id, Message("assistant", content="hi"), iteration=1)

    resumed = rebuild_transcript(store, info.id)
    assert isinstance(resumed, ResumedSession)
    assert [m.role for m in resumed.messages] == ["user", "assistant"]
    assert [m.content for m in resumed.messages] == ["hello", "hi"]
    assert resumed.session.id == info.id


def test_reasoning_comes_back_with_the_message(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(
        info.id, Message("assistant", content="answer", reasoning="thinking"), iteration=1
    )
    (message,) = rebuild_transcript(store, info.id).messages
    assert message.reasoning == "thinking"


def test_a_completed_tool_group_rebuilds_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(info.id, Message.user("read it"), iteration=1)
    turn = _ask(store, info.id, 1, ToolCall("c1", "read", '{"path": "a.py"}'))
    _answer(store, info.id, turn, _execution("c1"))
    store.append_turn(info.id, Message("assistant", content="it says one"), iteration=2)

    messages = rebuild_transcript(store, info.id).messages
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls == (ToolCall("c1", "read", '{"path": "a.py"}'),)
    assert messages[2].tool_call_id == "c1"
    assert messages[2].content == "1|one"


def test_a_dangling_assistant_call_is_dropped(tmp_path: Path) -> None:
    """Crashed before the results were written: the transcript must end clean."""
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(info.id, Message.user("read it"), iteration=1)
    store.append_turn(
        info.id, Message("assistant", tool_calls=(ToolCall("c1", "read", "{}"),)), iteration=1
    )

    resumed = rebuild_transcript(store, info.id)
    assert [m.role for m in resumed.messages] == ["user"]
    assert resumed.dropped > 0


def test_a_partially_answered_group_is_dropped_whole(tmp_path: Path) -> None:
    """Two calls, one result: keeping the group would be a 400 on the next request."""
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(info.id, Message.user("read both"), iteration=1)
    turn = _ask(store, info.id, 1, ToolCall("c1", "read", "{}"), ToolCall("c2", "read", "{}"))
    store.append_tool_call(info.id, turn, _execution("c1"))
    store.append_turn(info.id, Message.tool_result("c1", "1|one"), iteration=1)

    resumed = rebuild_transcript(store, info.id)
    assert [m.role for m in resumed.messages] == ["user"]
    assert resumed.dropped == 2  # the assistant row and the one result that landed


def test_a_group_whose_results_come_back_out_of_order_is_still_valid(tmp_path: Path) -> None:
    """The store keeps rows in insertion order; the wire needs results in call order."""
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    turn = _ask(store, info.id, 1, ToolCall("c1", "read", "{}"), ToolCall("c2", "grep", "{}"))
    store.append_tool_call(info.id, turn, _execution("c2", name="grep", observation="hit"))
    store.append_tool_call(info.id, turn, _execution("c1", observation="1|one"))
    store.append_turn(info.id, Message.tool_result("c2", "hit"), iteration=1)
    store.append_turn(info.id, Message.tool_result("c1", "1|one"), iteration=1)

    messages = rebuild_transcript(store, info.id).messages
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c1", "c2"]


def test_resuming_an_unknown_session_is_not_silent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert rebuild_transcript(store, "does-not-exist") is None


def test_an_empty_session_resumes_to_nothing_rather_than_failing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    resumed = rebuild_transcript(store, info.id)
    assert resumed is not None
    assert resumed.messages == []
    assert resumed.dropped == 0


def test_the_rebuilt_history_carries_the_counts_the_tui_prints(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="p", model="m")
    store.append_turn(info.id, Message.user("one"), iteration=1)
    store.append_turn(info.id, Message("assistant", content="two"), iteration=1)
    resumed = rebuild_transcript(store, info.id)
    assert resumed.turns == 2
    assert resumed.session.turns == 2

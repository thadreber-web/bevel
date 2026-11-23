"""The todo list in the loop: prompt guidance, session wiring, and the row landing.

The tool tests construct a :class:`ToolContext` directly, so they prove the tool
works but not that the loop hands it a session. Three things can break between
them and none of them show up there: the store never reaching the context, the
session id never being assigned to it, or the guidance being appended when no
store is attached. All are asserted on the database rather than on a mock -- the
row landing under the right session is the only thing the user cares about.
"""

from __future__ import annotations

from pathlib import Path

from bevel.config.schema import PolicyConfig
from bevel.core.agent import Agent
from bevel.core.session.resume import rebuild_transcript
from bevel.core.session.store import SessionStore, Todo, TodoStatus, TodoStore
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

_TASK = "fix the parser"


def _stores(tmp_path: Path) -> tuple[SessionStore, TodoStore]:
    """Both stores on one file, which is what ``run_tui`` does."""
    db = tmp_path / "sessions.db"
    return SessionStore(db), TodoStore(db)


def _todo_turn(*items: tuple[str, str]) -> list[ScriptedTurn]:
    todos = [{"content": content, "status": status} for content, status in items]
    call = tool_call("c1", "todo", {"todos": todos})
    return [ScriptedTurn(content="", tool_calls=(call,)), ScriptedTurn(content="done")]


def _policy_config(tmp_path: Path) -> PolicyConfig:
    """Keep the decision log out of the user's real ``~/.cache/bevel``."""
    return PolicyConfig(db_path=tmp_path / "policy.db")


async def test_a_scripted_todo_call_lands_in_the_database(tmp_path: Path) -> None:
    """The whole path: model asks, loop executes, row is on disk under this session."""
    store, todos = _stores(tmp_path)
    provider = MockProvider(turns=_todo_turn(("read the parser", "done"), ("fix it", "pending")))
    agent = Agent(
        provider, tmp_path, store=store, todos=todos, policy_config=_policy_config(tmp_path)
    )

    result = await agent.run(_TASK)

    assert result.ok, result.error
    assert agent.session_id is not None
    reopened = TodoStore(store.db_path)
    reopened.session_id = agent.session_id
    assert [(t.content, str(t.status)) for t in reopened.list_todos()] == [
        ("read the parser", "done"),
        ("fix it", "pending"),
    ]
    reopened.close()
    store.close()
    todos.close()


async def test_the_engine_lets_it_through_as_a_read(tmp_path: Path) -> None:
    """``READ`` has to survive the engine, or every progress update prompts.

    The tool tests assert the declaration; the engine tests construct an engine
    directly. Neither covers the one path the loop takes -- ``Agent`` builds its
    own engine and evaluates the call through it. A tier that drifted to
    ``WRITE_WORKSPACE`` would leave the decision log saying so, and in default
    mode the call would be refused with no approver installed, while every
    tool-level test stayed green.
    """
    store, todos = _stores(tmp_path)
    provider = MockProvider(turns=_todo_turn(("do the thing", "in_progress")))
    agent = Agent(
        provider, tmp_path, store=store, todos=todos, policy_config=_policy_config(tmp_path)
    )

    result = await agent.run(_TASK)

    (execution,) = [e for event in result.events for e in getattr(event, "executions", ())]
    assert execution.outcome.value == "ok", f"{execution.outcome}: {execution.observation}"
    (decision,) = agent._policy.decisions()
    assert decision.rule == "tier:read", (
        f"todo was allowed by {decision.rule!r}, not as a read -- an approval prompt "
        "would now sit on every progress update"
    )
    todos.session_id = agent.session_id
    assert [t.content for t in todos.list_todos()] == ["do the thing"]
    store.close()
    todos.close()


async def test_the_prompt_tells_the_model_to_use_it_only_when_a_store_is_attached(
    tmp_path: Path,
) -> None:
    """The registry offers ``todo`` either way, so silence is the only off-switch.

    Without guidance a small model answers a five-step task in one unstructured
    paragraph and the user sees no plan. With guidance and *no* store, the model
    spends calls on a tool that can only fail.
    """
    _, todos = _stores(tmp_path)
    off = await Agent(MockProvider(turns=[ScriptedTurn(content="hi")]), tmp_path).run(_TASK)
    on = await Agent(MockProvider(turns=[ScriptedTurn(content="hi")]), tmp_path, todos=todos).run(
        _TASK
    )

    off_prompt = off.messages[0].content.lower()
    on_prompt = on.messages[0].content.lower()
    # Not the bare word "todo": the project map lists ``tools/todo.py`` either
    # way. What rides on the flag is the guidance block.
    assert "task tracking is on" not in off_prompt, "the prompt offers a tool that can only fail"
    assert "task tracking is on" in on_prompt, "the model is never told to keep a plan"
    # The wording that matters is the replace semantics: a model that appends
    # produces a list that quietly keeps growing tasks it abandoned.
    assert "replaces what is stored, it does not append" in on_prompt
    todos.close()


async def test_a_resumed_session_writes_its_todos_under_its_own_id(tmp_path: Path) -> None:
    """Resume is the path where the id is not created here, so it is easy to miss.

    ``_open_session`` returns early on resume, and the assignment has to happen
    anyway -- otherwise the tool refuses with "no session is open" for the one
    run that has a session id from the very start.
    """
    store, todos = _stores(tmp_path)
    info = store.create_session(workspace=tmp_path, provider="mock", model="mock")
    todos.session_id = None

    provider = MockProvider(turns=_todo_turn(("continue the work", "in_progress")))
    agent = Agent(
        provider,
        tmp_path,
        store=store,
        resume=rebuild_transcript(store, info.id),
        todos=todos,
        policy_config=_policy_config(tmp_path),
    )
    result = await agent.run("keep going")

    assert result.ok, result.error
    assert todos.session_id == info.id
    assert [t.content for t in todos.list_todos()] == ["continue the work"]
    store.close()
    todos.close()


async def test_a_todo_store_with_no_session_store_refuses_rather_than_orphans_rows(
    tmp_path: Path,
) -> None:
    """No SessionStore means no session row, and the table has a foreign key to it.

    The honest answer is a refusal, not a row that no session can read back. If
    the assignment ever moved inside the ``SessionStore`` branch of
    ``_open_session`` this stays green by accident, so it also asserts the store
    was actually consulted -- the observation is the tool's own, not the
    validator's.
    """
    _, todos = _stores(tmp_path)
    provider = MockProvider(turns=_todo_turn(("work without persistence", "pending")))
    agent = Agent(provider, tmp_path, todos=todos, policy_config=_policy_config(tmp_path))

    result = await agent.run(_TASK)

    assert result.ok, "a refused tool call must not fail the run"
    assert agent.session_id is None
    (execution,) = [e for event in result.events for e in getattr(event, "executions", ())]
    assert execution.outcome.value == "failed"
    assert "session" in execution.observation.lower()
    assert not todos.db_path.exists(), "a refused call must not create a database"
    todos.close()


async def test_a_todo_call_with_no_store_fails_the_call_not_the_run(tmp_path: Path) -> None:
    """The registry offers ``todo`` with no store attached; the model may still call it.

    The observation has to say why nothing happened, and the run has to finish
    rather than abort -- a refused tool call is a normal event in a transcript.
    """
    provider = MockProvider(turns=_todo_turn(("anything", "pending")))
    agent = Agent(provider, tmp_path, policy_config=_policy_config(tmp_path))

    result = await agent.run(_TASK)

    (execution,) = [e for event in result.events for e in getattr(event, "executions", ())]
    assert execution.outcome.value == "failed"
    assert "todo" in execution.observation
    assert "store" in execution.observation.lower()
    assert result.ok, "a refused tool call must not fail the run"


async def test_a_second_run_in_the_same_session_replaces_the_list(tmp_path: Path) -> None:
    """Two runs, one session: the second list is the list, not six rows.

    This is the append bug in loop clothing -- the shape the TUI actually drives,
    where a second task would otherwise leave the first task's rows on screen.
    """
    store, todos = _stores(tmp_path)
    first = MockProvider(turns=_todo_turn(("a", "done"), ("b", "pending"), ("c", "pending")))
    agent = Agent(first, tmp_path, store=store, todos=todos, policy_config=_policy_config(tmp_path))
    await agent.run(_TASK)
    session_id = agent.session_id

    second = MockProvider(turns=_todo_turn(("b", "done")))
    again = Agent(
        second,
        tmp_path,
        store=store,
        resume=rebuild_transcript(store, session_id) if session_id else None,
        todos=todos,
        policy_config=_policy_config(tmp_path),
    )
    await again.run("and now?")

    assert again.session_id == session_id
    assert [t.content for t in todos.list_todos()] == ["b"]
    store.close()
    todos.close()


def test_the_stored_rows_are_the_ones_the_tool_declares(tmp_path: Path) -> None:
    """A guard against the enum and the schema drifting apart.

    The tool builds its ``enum`` from ``TodoStatus`` and the store parses rows
    back through it, so a status added to one and not the other is a crash on
    read rather than a compile error.
    """
    assert [str(status) for status in TodoStatus] == [
        "pending",
        "in_progress",
        "done",
    ]
    assert Todo("x", TodoStatus.DONE).status is TodoStatus.DONE


async def test_the_tool_list_the_model_is_offered_includes_todo(tmp_path: Path) -> None:
    """The loop takes its tools from the registry; a filtered list would hide it."""
    agent = Agent(MockProvider(turns=[ScriptedTurn(content="hi")]), tmp_path)
    assert "todo" in [tool.spec.name for tool in agent.tools]

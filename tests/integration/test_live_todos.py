"""The todo list against a real endpoint.

``tests/unit/test_tools_todo.py`` proves the store replaces and scopes, and
``tests/unit/test_agent_todos.py`` proves the loop wires the store in. Neither
proves the thing that would actually embarrass this feature: that a model
*deciding for itself* what its plan is can fill in an array-of-objects schema,
and that what it wrote is still on disk in another process.

So these do the real thing. The deterministic half -- the rows are in the file,
read through a second connection -- is asserted first and never retried. Only
then does a model get asked, and the assertion is on the **database**, never on
the transcript: a model can describe a plan it never stored, and a loop can
print an observation it never wrote.

Cost: two or three real generations per test, marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from bevel.core.agent import Agent, AgentEventType, AgentResult, Budgets, StopReason, ToolExecution
from bevel.core.session.resume import rebuild_transcript
from bevel.core.session.store import SessionStore, Todo, TodoStatus, TodoStore
from bevel.providers.base import GenOptions, ModelProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="module"),
]

#: Three tasks the model is given and has to store. They share no words with the
#: instruction, so a row that came from the prompt rather than from the model's
#: own call would be visible in the assertions below.
TASKS: Final[tuple[str, ...]] = (
    "read the tokenizer in providers/openai_compat.py",
    "add a regression test for the truncated stream",
    "run the unit suite and report the count",
)

#: Words that identify each task above, in the same order. A model rephrases,
#: so the live assertions match on these rather than on the exact sentences.
KEYWORDS: Final[tuple[tuple[str, ...], ...]] = (
    ("tokenizer", "openai_compat"),
    ("truncated", "stream"),
    ("unit", "count"),
)

#: Deterministic sampling, as in the other live tests.
GEN: Final[GenOptions] = GenOptions(temperature=0.0, max_tokens=512)


def _requires_tools(endpoint: ModelProvider) -> None:
    if not endpoint.capabilities.supports_tools:
        pytest.skip(f"{endpoint.model!r} does not support tool calling (probed; see bevel doctor)")


def _executions(result: AgentResult) -> list[ToolExecution]:
    return [
        execution
        for event in result.events
        if event.type is AgentEventType.TOOL_COMPLETE
        for execution in event.executions
    ]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A database path that is *not* the user's real one."""
    return tmp_path / "share" / "bevel" / "sessions.db"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


# ---------------------------------------------------------------- the reopen test


async def test_todos_written_by_one_store_are_read_by_another(
    workspace: Path, db_path: Path
) -> None:
    """Close the database, reopen it from a second store, and the list is still there.

    No model, so this half is fully deterministic. The reading store is a fresh
    object on a fresh connection to the same file -- the writing one is closed
    and dropped -- so nothing can be coming from process memory.
    """
    sessions = SessionStore(db_path)
    try:
        info = sessions.create_session(workspace=workspace, provider="integration", model="probe")
    finally:
        sessions.close()

    writer = TodoStore(db_path)
    try:
        writer.session_id = info.id
        writer.replace(
            [
                Todo(content=TASKS[0], status=TodoStatus.DONE),
                Todo(content=TASKS[1], status=TodoStatus.IN_PROGRESS),
                Todo(content=TASKS[2], status=TodoStatus.PENDING),
            ]
        )
    finally:
        writer.close()

    reader = TodoStore(db_path)
    try:
        reader.session_id = info.id
        rows = reader.list_todos()
        assert [(todo.content, str(todo.status)) for todo in rows] == [
            (TASKS[0], "done"),
            (TASKS[1], "in_progress"),
            (TASKS[2], "pending"),
        ], "the list did not survive the file"
    finally:
        reader.close()

    # A second session on the same file sees nothing: the rows belong to the one
    # that wrote them.
    other = SessionStore(db_path)
    try:
        second = other.create_session(workspace=workspace, provider="integration", model="probe").id
    finally:
        other.close()
    stranger = TodoStore(db_path)
    try:
        stranger.session_id = second
        assert stranger.list_todos() == (), "one session could read another's plan"
    finally:
        stranger.close()


async def test_the_model_can_call_todo_itself(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """A model filling in the array-of-objects schema, with the rows checked on disk.

    The unit tests script the call, so they cannot show that a small model
    produces a valid nested array under ``additionalProperties: false`` -- one
    invented key, or a status spelled loosely, and the validator refuses the
    whole call. And the assertion is deliberately on the database rather than on
    ``result.final_text``: prose describing a plan is not a plan.
    """
    _requires_tools(endpoint)

    sessions = SessionStore(db_path)
    todos = TodoStore(db_path)
    try:
        agent = Agent(
            endpoint,
            workspace,
            budgets=Budgets(max_iterations=4),
            store=sessions,
            todos=todos,
            options=GEN,
        )
        result = await agent.run(
            "Plan the work described here using the todo tool, then say 'planned'. Do not "
            "start the work. The steps are: " + "; ".join(TASKS) + "."
        )
        assert result.stop_reason is StopReason.COMPLETED, result.error

        calls = [e for e in _executions(result) if e.call.name == "todo"]
        assert calls, (
            f"the model never called todo; it called "
            f"{[e.call.name for e in _executions(result)] or 'nothing'}"
        )
        assert any(e.succeeded for e in calls), (
            "todo was called and refused; the schema is one this model cannot satisfy: "
            + "; ".join(e.result.error or "" for e in calls if e.result)
        )

        assert agent.session_id is not None
        reopened = TodoStore(db_path)
        try:
            reopened.session_id = agent.session_id
            contents = [todo.content for todo in reopened.list_todos()]
        finally:
            reopened.close()
        assert contents, "the tool reported success but no row reached the database"
        # The model writes the phrasing, so the check is on content rather than
        # on the exact strings: one row per task it was given, in the order it
        # was given, each recognisably the right one. A model that invented a
        # plan, dropped a step, or padded the list fails here.
        assert len(contents) == len(TASKS), contents
        for content, keywords in zip(contents, KEYWORDS, strict=True):
            lowered = content.lower()
            assert any(word in lowered for word in keywords), (
                f"row {content!r} does not look like the task it was given: "
                f"{', '.join(keywords)!r} expected"
            )
    finally:
        sessions.close()
        todos.close()


async def test_a_second_run_replaces_the_list_rather_than_growing_it(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """Two runs, one session, and the list is the second one.

    This is the append bug against a real model. A store that appended would
    leave five rows here and every scripted test would still be green, because
    the scripted tests only ever check that a row exists.
    """
    _requires_tools(endpoint)

    sessions = SessionStore(db_path)
    todos = TodoStore(db_path)
    try:
        first = Agent(
            endpoint,
            workspace,
            budgets=Budgets(max_iterations=4),
            store=sessions,
            todos=todos,
            options=GEN,
        )
        first_result = await first.run(
            "Record a plan with the todo tool, then say 'ok'. Steps: " + "; ".join(TASKS) + "."
        )
        first_calls = [e for e in _executions(first_result) if e.call.name == "todo"]
        assert any(e.succeeded for e in first_calls), (
            "the first run never wrote a list, so there was nothing to replace"
        )
        assert first.session_id is not None
        session_id = first.session_id

        second = Agent(
            endpoint,
            workspace,
            budgets=Budgets(max_iterations=4),
            store=sessions,
            resume=rebuild_transcript(sessions, session_id),
            todos=todos,
            options=GEN,
        )
        second_result = await second.run(
            "Using the todo tool, record a plan with exactly one step, then say 'ok'. "
            "The step is: 'write the changelog'."
        )
        assert second.session_id == session_id, "the second run started a new session"
        second_calls = [e for e in _executions(second_result) if e.call.name == "todo"]
        assert any(e.succeeded for e in second_calls), (
            "the second run never rewrote the list, so this test proved nothing"
        )

        reopened = TodoStore(db_path)
        try:
            reopened.session_id = session_id
            contents = [todo.content for todo in reopened.list_todos()]
        finally:
            reopened.close()
        assert len(contents) == 1, f"the second run's list was merged with the first's: {contents}"
        assert "changelog" in contents[0].lower(), contents
    finally:
        sessions.close()
        todos.close()

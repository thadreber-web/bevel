"""Session persistence and resume against a real endpoint.

``tests/unit/test_agent_persist.py`` proves the loop writes what ``MockProvider``
was scripted to say, and that ``rebuild_transcript`` turns those rows back into a
requestable transcript. Neither proves the thing that matters to a user: that a
conversation with a model *deciding for itself* what to call survives closing the
process, and that the model then continues as if nothing happened.

So these tests do the real thing: run an agent against the server, close the
store, open a second ``SessionStore`` against the same file (a fresh connection --
nothing is left in memory), rebuild, and ask the model to recall a fact only the
first session contained. The fact is a random token that appears nowhere in any
prompt or instruction, so the answer can only come out of the rebuilt transcript.

Cost: three or four real generations per test, marked ``slow``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import pytest

from bevel.core.agent import Agent, AgentEventType, AgentResult, Budgets, StopReason, ToolExecution
from bevel.core.session.resume import rebuild_transcript
from bevel.core.session.store import SessionStore
from bevel.providers.base import GenOptions, ModelProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="module"),
]

#: A fact the model can only repeat by having been shown the earlier turn again.
#: It appears in no system prompt, no project map, and no other test.
FACT: Final[str] = "crimson-lantern-7731"

#: A second fact, for the run that also calls tools, so the stored observation can
#: be checked against the file on disk rather than against a scripted string.
SECRET: Final[str] = "zephyr-quartz-4417"

#: Deterministic sampling, as in the other live tests.
GEN: Final[GenOptions] = GenOptions(temperature=0.0, max_tokens=256)

#: How many times to re-ask the recall question. Temperature 0 is not bit-reproducible
#: on a batching server, so a small model occasionally answers around the fact instead
#: of with it. Re-asking is fair only because the deterministic assertion -- the fact
#: is present in the rebuilt transcript read back from disk -- is made first and is
#: never retried.
RECALL_ATTEMPTS: Final[int] = 3


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
def workspace(tmp_path: Path) -> Path:
    """A throwaway git repo holding the file the tool-calling session reads."""
    (tmp_path / "notes.txt").write_text(
        f"Meeting notes.\nThe access word is {SECRET}.\n",
        encoding="utf-8",
    )
    for argv in (
        ["init", "-q"],
        ["add", "notes.txt"],
        [
            "-c",
            "user.name=bevel",
            "-c",
            "user.email=bevel@localhost",
            "commit",
            "-q",
            "-m",
            "seed",
        ],
    ):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True, timeout=30)
    return tmp_path


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A database path that is *not* the user's real one."""
    return tmp_path / "share" / "bevel" / "sessions.db"


@pytest.fixture
def stored_workspace(workspace: Path) -> str:
    """What the store records for ``workspace``.

    Resolved here rather than in the test: ``Path.resolve`` is blocking I/O and the
    tests that compare against it are async.
    """
    return str(workspace.resolve())


# ------------------------------------------------------------------- the recall test


async def test_a_closed_session_reopens_and_the_model_remembers(
    endpoint: ModelProvider, workspace: Path, db_path: Path, stored_workspace: str
) -> None:
    """Close a session, reopen it from the file, and the history is still there.

    Three separate claims, each with its own assertion:

    1. the first run wrote its turns to the database (checked by row count, not by
       trusting the in-memory transcript),
    2. a *second* ``SessionStore`` -- a new connection, so nothing can be coming
       from memory -- rebuilds a transcript that literally contains the fact,
    3. a model shown that transcript produces the fact, which is the only part a
       unit test with ``MockProvider`` cannot establish.
    """
    store = SessionStore(db_path)
    try:
        agent = Agent(
            endpoint, workspace, store=store, budgets=Budgets(max_iterations=2), options=GEN
        )
        result = await agent.run(
            f"Remember this for later, I will ask you for it: the launch code is {FACT}. "
            "Reply with exactly: noted"
        )
        assert result.ok, result.error
        assert result.session_id, "a run with a store attached must report its session id"
        session_id = result.session_id

        # 1. The rows are in the file, and the session row accounts for them.
        info = store.get(session_id)
        assert info is not None
        assert info.workspace == stored_workspace
        assert info.turns >= 2, f"expected a user turn and a reply, got {info.turns}"
        assert info.prompt_tokens > 0, f"the session row recorded no prompt usage: {info}"
        assert info.total_tokens > 0, f"the session row recorded no usage: {info}"
    finally:
        store.close()

    # 2. A fresh connection to the same path. The first store is closed and its
    # Python object is gone, so anything read here came off the disk.
    reopened = SessionStore(db_path)
    try:
        resumed = rebuild_transcript(reopened, session_id)
        assert resumed is not None, "the session vanished from the file"
        assert resumed.dropped == 0, (
            f"resume discarded {resumed.dropped} stored rows from a cleanly closed session"
        )
        wire = "\n".join(f"{message.role}: {message.content}" for message in resumed.messages)
        assert FACT in wire, "the fact never made it into the rebuilt transcript"
        assert resumed.session.id == session_id
        assert len(resumed.stored) == len(resumed.messages)

        # 3. The model, shown that transcript, still knows what was said.
        async def ask() -> AgentResult:
            continuation = Agent(
                endpoint, workspace, resume=resumed, budgets=Budgets(max_iterations=2)
            )
            return await continuation.run(
                "What launch code did I tell you earlier? Answer with just the code."
            )

        last: AgentResult | None = None
        for _ in range(RECALL_ATTEMPTS):
            last = await ask()
            if FACT in last.final_text:
                break
        assert last is not None
        assert last.stop_reason is StopReason.COMPLETED, last.error
        assert FACT in last.final_text, (
            f"the resumed model did not remember; it said: {last.final_text!r}"
        )
    finally:
        reopened.close()


async def test_a_resumed_run_writes_to_the_same_session(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """Continuing a session appends to it rather than starting a parallel one.

    ``bevel resume`` twice in a row must not scatter one conversation across three
    session rows, or ``bevel sessions`` becomes a list of fragments.
    """
    store = SessionStore(db_path)
    try:
        first = Agent(endpoint, workspace, store=store, budgets=Budgets(max_iterations=1))
        one = await first.run("Reply with exactly: one")
        assert one.ok, one.error
        assert one.session_id
        before = len(store.turns(one.session_id))

        resumed = rebuild_transcript(store, one.session_id)
        assert resumed is not None
        second = Agent(
            endpoint,
            workspace,
            store=store,
            resume=resumed,
            budgets=Budgets(max_iterations=1),
        )
        two = await second.run("Reply with exactly: two")
        assert two.ok, two.error
        assert two.session_id == one.session_id, "the resumed run opened a new session"

        after = store.turns(one.session_id)
        assert len(after) > before, "the resumed run added nothing to the session"
        assert after[-1].content.strip().lower().startswith("two"), after[-1].content
    finally:
        store.close()


# ------------------------------------------------- everything really is in the file


async def test_every_turn_tool_call_and_usage_row_lands_in_the_database(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """The reopened transcript is read back from the store, not held in memory.

    A run that calls a tool must leave: one session row, one turn row per message,
    one tool-call row per execution carrying the real observation, and one usage row
    per iteration the server reported. Each is compared against what the loop
    actually did, so a write path that silently no-ops on one of the four tables
    fails here.
    """
    _requires_tools(endpoint)
    store = SessionStore(db_path)
    try:
        agent = Agent(endpoint, workspace, store=store, budgets=Budgets(max_iterations=6))
        result = await agent.run(
            "Read the file notes.txt and tell me the access word it contains. "
            "Answer with just that word."
        )
        assert result.stop_reason is StopReason.COMPLETED, result.error
        assert result.session_id
        session_id = result.session_id

        executions = _executions(result)
        assert executions, "the model never called a tool, so this test proved nothing"

        # The system prompt is the one message that is deliberately *not* stored: a
        # resumed session rebuilds it so the project map is current rather than the
        # one from when the session was opened. Everything else must be on disk.
        assert result.messages[0].role == "system"
        expected = list(result.messages[1:])
        turns = store.turns(session_id)
        assert len(turns) == len(expected), (
            f"{len(expected)} messages in the loop, {len(turns)} rows in the file"
        )
        assert [turn.role for turn in turns] == [message.role for message in expected]
        assert [turn.content for turn in turns] == [message.content for message in expected], (
            "a stored message differs from what the loop held"
        )
        assert [turn.reasoning for turn in turns] == [message.reasoning for message in expected], (
            "a stored message lost its reasoning"
        )

        stored_calls = [call for turn in turns for call in turn.tool_calls]
        assert len(stored_calls) == len(executions)
        for stored, execution in zip(stored_calls, executions, strict=True):
            assert stored.call_id == execution.call.id
            assert stored.name == execution.call.name
            assert stored.arguments == execution.call.arguments
            assert stored.outcome == execution.outcome.value
            if execution.result is not None:
                assert stored.observation == execution.result.content, (
                    f"{stored.name}: the observation the model saw is not in the file"
                )
        assert any(SECRET in call.observation for call in stored_calls), (
            "the read observation was never stored, so a reopened session would "
            "show a tool call with nothing it learned"
        )

        rows = store.usage_rows(session_id)
        assert rows, "no usage rows were written; `bevel sessions` would list zeros"
        assert sum(row.total_tokens for row in rows) == result.usage.total_tokens, (
            f"the usage rows sum to {sum(row.total_tokens for row in rows)}, the run "
            f"reported {result.usage.total_tokens}"
        )
        info = store.get(session_id)
        assert info is not None
        assert info.total_tokens == result.usage.total_tokens
        assert info.turns == len(turns)
    finally:
        store.close()


async def test_a_reopened_session_lists_in_the_index(
    endpoint: ModelProvider, workspace: Path, db_path: Path, stored_workspace: str
) -> None:
    """``bevel sessions`` and bare ``bevel resume`` work off the same rows.

    Both read through ``list_sessions`` / ``latest_session_id``; if either key were
    written wrong the TUI would open on an empty list while the session existed.
    """
    store = SessionStore(db_path)
    try:
        agent = Agent(endpoint, workspace, store=store, budgets=Budgets(max_iterations=1))
        result = await agent.run("Reply with exactly: listed")
        assert result.ok, result.error
        assert result.session_id

        listed = store.list_sessions()
        assert [session.id for session in listed] == [result.session_id]
        assert listed[0].workspace == stored_workspace
        assert listed[0].model == endpoint.model
        assert listed[0].started_at
        assert listed[0].last_activity_at
        assert store.latest_session_id(workspace) == result.session_id
        assert store.latest_session_id(workspace.parent) is None, (
            "an unrelated workspace resolved to someone else's session"
        )
        assert store.resolve(result.session_id[:6]).id == result.session_id
    finally:
        store.close()

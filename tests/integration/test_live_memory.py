"""Long-term memory against a real endpoint.

``tests/unit/test_memory.py`` proves the store and the ranking agree with a
transcript someone wrote by hand, and ``tests/unit/test_agent_memory.py`` proves
the loop injects the block where compaction can reach it. Neither proves the
thing a user feels: that a memory written in one process is still there in
another, and that a model *deciding for itself* what to ask memory can use the
answer.

So these tests do the real thing: write a memory, close the store, open a
**second** ``MemoryStore`` against the same file (a fresh connection, so nothing
can be coming from memory), and ask a model for a random token only the first
session ever contained. The token appears in no system prompt, no project map
and no other test, so it can only arrive out of the database.

Vectors are deliberately not exercised: the target endpoints are vLLM and
llama.cpp servers that mostly do not expose ``/embeddings``, and a memory test
that silently degraded to lexical recall would prove nothing either way. The
lexical path is the one that must always work, so it is the one asserted here.

Cost: two or three real generations per test, marked ``slow``.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Final

import pytest

from bevel.config.schema import MemoryConfig
from bevel.core.agent import Agent, AgentEventType, AgentResult, Budgets, StopReason, ToolExecution
from bevel.core.memory.recall import recall
from bevel.core.memory.store import MemoryStore, MemoryType, repo_scope
from bevel.core.tools.memory import memory_tools
from bevel.providers.base import ModelProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.asyncio(loop_scope="module"),
]

#: A token the model can only produce by having the memory in front of it.
TOKEN: Final[str] = f"zq{secrets.token_hex(3)}"

#: The memory and the question that asks for it. They share every word except
#: the token, so FTS5 matches on the words and the token can only come from the
#: stored row -- a query that shared nothing would recall nothing, and a store
#: that returned everything would pass a test that shared the token too.
MEMORY: Final[str] = f"The deploy passphrase for the bevel widget service is {TOKEN}."
QUESTION: Final[str] = (
    "What is the deploy passphrase for the bevel widget service? Answer with just the passphrase."
)

#: Temperature 0 is not bit-reproducible on a batching server, so a small model
#: occasionally answers around the token instead of with it. Re-asking is fair
#: only because the deterministic assertion -- the token is in the recall result
#: read off the disk -- is made first and is never retried.
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
def db_path(tmp_path: Path) -> Path:
    """A database path that is *not* the user's real one."""
    return tmp_path / "share" / "bevel" / "memories.db"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


# ---------------------------------------------------------------- the reopen test


async def test_a_memory_written_by_one_store_is_recalled_by_another(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """Close the database, reopen it from a second store, and the memory is still there.

    Three claims, each with its own assertion:

    1. the first store wrote the row to the file (checked through a *second*
       connection, not through the object that wrote it),
    2. recall from that fresh connection returns the token and says "lexical",
       because nothing here opted in to embeddings,
    3. a model shown the injected block produces the token, which is the part a
       unit test with ``MockProvider`` cannot establish.
    """
    store = MemoryStore(db_path, config=MemoryConfig(enabled=True))
    try:
        added = store.add(content=MEMORY, type=MemoryType.FACT, scope=await repo_scope(workspace))
        assert added.id >= 1
        assert added.content == MEMORY
    finally:
        store.close()

    # A fresh connection to the same path. The first store is closed and its
    # Python object is gone, so anything read here came off the disk.
    reopened = MemoryStore(db_path, config=MemoryConfig(enabled=True))
    try:
        # 1 + 2. The deterministic half: no model, no retries, no wiggle room.
        result = await recall(reopened, QUESTION, workspace=workspace)
        assert result.memories, "the memory vanished from the file"
        assert result.mode == "lexical", (
            f"recall reported {result.mode!r} without anyone opting in to embeddings"
        )
        assert TOKEN in result.memories[0].memory.content
        assert reopened.get(added.id) is not None, "the row is unreadable by id"

        # 3. The model, shown the block the loop injected, still knows the token.
        async def ask() -> AgentResult:
            agent = Agent(
                endpoint,
                workspace,
                budgets=Budgets(max_iterations=2),
                tools=[],
                memory=reopened,
            )
            return await agent.run(QUESTION)

        last: AgentResult | None = None
        for _ in range(RECALL_ATTEMPTS):
            last = await ask()
            if TOKEN in last.final_text:
                break
        assert last is not None
        assert last.stop_reason is StopReason.COMPLETED, last.error
        assert TOKEN in last.final_text, (
            f"the model did not use the injected memory; it said: {last.final_text!r}"
        )
    finally:
        reopened.close()


async def test_the_model_can_call_recall_itself(
    endpoint: ModelProvider, workspace: Path, db_path: Path
) -> None:
    """The ``recall`` tool is real: a model asking its own question gets the answer.

    The injection test above hands the model the memory unprompted. This one
    makes the model ask, which additionally proves the tool's schema is one a
    small model can fill in and that the observation reaches the transcript --
    the assertion is that the token appears in the *tool result*, not only in
    the final prose, so a loop that dropped observations would fail.
    """
    _requires_tools(endpoint)
    store = MemoryStore(db_path, config=MemoryConfig(enabled=True))
    try:
        store.add(content=MEMORY, type=MemoryType.FACT, scope=await repo_scope(workspace))
        remember, recall_tool = memory_tools(store, workspace)
        agent = Agent(
            endpoint,
            workspace,
            budgets=Budgets(max_iterations=4),
            tools=[recall_tool, remember],
            memory=store,
        )
        result = await agent.run(
            "Before answering, use the recall tool to look up the deploy passphrase for "
            "the bevel widget service. Then answer with just the passphrase."
        )
        assert result.stop_reason is StopReason.COMPLETED, result.error

        executions = _executions(result)
        calls = [e for e in executions if e.call.name == "recall"]
        assert calls, (
            f"the model never called recall; it called "
            f"{[e.call.name for e in executions] or 'nothing'}"
        )
        assert any(e.result and TOKEN in e.result.content for e in calls), (
            "recall ran but its observation did not contain the memory, so the model "
            "could not have read it"
        )
        assert any(e.result and e.result.ok for e in calls), "every recall call failed"
    finally:
        store.close()

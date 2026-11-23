"""Long-term memory in the opening context.

The orient step injects recalled memories, and the two things that matter about
it are *where* the message lands and what happens when memory is off. Position
matters because :mod:`bevel.core.context.compact` never evicts ``messages[0]``:
a memory block that became the system message could never be dropped, and a
stale one is exactly what compaction exists to remove.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import MemoryConfig
from bevel.core.agent import Agent
from bevel.core.memory.store import MemoryStore, MemoryType
from bevel.core.policy import allow_all_policy
from bevel.providers.base import Capabilities
from bevel.providers.mock import MockProvider, ScriptedTurn

#: Shared by the query and the memory, so FTS5 matches; the other memory shares
#: no words with either and must stay out.
QUERY = "how does bevel handle auth middleware"
MATCHED = "the auth middleware broke on empty tokens"
UNMATCHED = "the caching layer indexes documents nightly"


def _store(tmp_path: Path, *contents: str, **config: Any) -> MemoryStore:
    store = MemoryStore(tmp_path / "memories.db", config=MemoryConfig(**config))
    for content in contents:
        store.add(content=content, type=MemoryType.FACT, scope="global")
    return store


def _agent(workspace: Path, provider: MockProvider, memory: MemoryStore | None) -> Agent:
    return Agent(provider, workspace, policy=allow_all_policy, tools=[], memory=memory)


async def test_recalled_memory_is_in_the_opening_request(tmp_path: Path) -> None:
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    await _agent(tmp_path, provider, _store(tmp_path, MATCHED, UNMATCHED)).run(QUERY)

    sent = provider.requests[0]["messages"]
    assert isinstance(sent, list)
    text = "\n".join(message.content for message in sent)
    assert MATCHED in text
    assert UNMATCHED not in text, "an irrelevant memory reached the model"


async def test_the_memory_message_is_not_messages_zero(tmp_path: Path) -> None:
    """Index 1, below the system message, so compaction can still evict it."""
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    result = await _agent(tmp_path, provider, _store(tmp_path, MATCHED)).run(QUERY)

    assert result.messages[0].role == "system"
    assert result.messages[1].role == "user"
    assert MATCHED in result.messages[1].content
    assert result.messages[2].content == QUERY, "the task must stay last"


async def test_no_memory_store_leaves_the_transcript_untouched(tmp_path: Path) -> None:
    """``memory=None`` must be exactly the pre-m09a loop, not an empty block."""
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    without = await _agent(tmp_path, provider, None).run(QUERY)
    with_empty = await _agent(tmp_path, provider, _store(tmp_path)).run(QUERY)

    assert [m.content for m in without.messages] == [m.content for m in with_empty.messages]
    assert all("bevel recall" not in m.content for m in without.messages)


async def test_a_store_that_matches_nothing_injects_nothing(tmp_path: Path) -> None:
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    result = await _agent(tmp_path, provider, _store(tmp_path, UNMATCHED)).run(QUERY)

    assert all("bevel recall" not in m.content for m in result.messages)


async def test_a_broken_recall_does_not_cost_the_user_their_reply(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Memory is an enhancement. A corrupt database must not swallow the answer."""
    provider = MockProvider(turns=[ScriptedTurn(content="Hello!")])
    store = _store(tmp_path, MATCHED)
    store.close()
    store.db_path.unlink()
    store.db_path.write_bytes(b"not a database")

    with caplog.at_level("ERROR", logger="bevel.agent"):
        result = await _agent(tmp_path, provider, store).run(QUERY)
    assert result.ok
    assert result.final_text == "Hello!"
    assert "recall" in caplog.text, "the failure was swallowed rather than reported"


async def test_vectors_opt_in_binds_the_provider_as_the_embedder(tmp_path: Path) -> None:
    """Only the run knows what endpoint it is talking to; the store outlives it."""
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    store = _store(tmp_path, MATCHED, vectors=True)
    assert store.embedder is None

    await _agent(tmp_path, provider, store).run(QUERY)
    assert store.embedder is provider


async def test_vectors_opt_out_leaves_the_embedder_alone(tmp_path: Path) -> None:
    """Without the opt-in, nothing may contact /embeddings on the user's behalf."""
    provider = MockProvider(
        turns=[ScriptedTurn(content="hi")],
        capabilities=Capabilities(supports_embeddings=True),
    )
    store = _store(tmp_path, MATCHED)

    await _agent(tmp_path, provider, store).run(QUERY)
    assert store.embedder is None

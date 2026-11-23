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
from bevel.core.memory.store import MemoryStore, MemoryType, repo_scope
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


async def test_a_memory_stored_for_the_workspace_is_injected(tmp_path: Path) -> None:
    """The opening context looks in the *workspace's* scope, not the process's.

    ``remember`` files a memory under the agent's workspace, so injection has to
    read from the same place. Deriving the scope from ``Path.cwd()`` instead
    passes every global-scope test and silently finds nothing the moment bevel
    is started from anywhere other than the project it is working on -- which is
    the normal case, since the TUI is launched from a shell in some other folder.
    """
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    store = MemoryStore(tmp_path / "memories.db", config=MemoryConfig(enabled=True))
    scope = await repo_scope(tmp_path)
    assert not scope.startswith("global"), "the fixture must exercise the per-repo tier"
    store.add(content=MATCHED, type=MemoryType.FACT, scope=scope)

    await _agent(tmp_path, provider, store).run(QUERY)

    sent = provider.requests[0]["messages"]
    assert isinstance(sent, list)
    text = "\n".join(message.content for message in sent)
    assert MATCHED in text, "a memory filed under the workspace scope was never recalled"


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

    # From index 1 on, an attached-but-empty store must change nothing. The
    # system message is allowed to differ: it names the memory tools when a
    # store is attached, which is the point of the store being attached.
    without_rest = [m.content for m in without.messages[1:]]
    with_empty_rest = [m.content for m in with_empty.messages[1:]]
    assert without_rest == with_empty_rest
    assert all("bevel recall" not in m.content for m in without.messages)


async def test_the_prompt_names_the_memory_tools_only_when_memory_is_on(
    tmp_path: Path,
) -> None:
    """The model reaches for ``remember`` on its own, so say when it is available.

    Left to the tool descriptions alone the model does discover ``remember`` --
    it called it unprompted against a real endpoint -- but it discovers it the
    same way when memory is switched off, and then spends a call on a tool that
    can only fail. So the guidance rides on the same flag as everything else
    about memory: a store attached means the prompt mentions the tools, no
    store means the prompt is the plain one.
    """
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    off = await _agent(tmp_path, provider, None).run(QUERY)
    on = await _agent(tmp_path, provider, _store(tmp_path, MATCHED)).run(QUERY)

    off_prompt = off.messages[0].content.lower()
    on_prompt = on.messages[0].content.lower()
    assert "remember" not in off_prompt, "the prompt offers memory tools that cannot be used"
    assert "recall" not in off_prompt, "the prompt offers memory tools that cannot be used"
    assert "remember" in on_prompt, "the model is never told it can store a memory"
    assert "recall" in on_prompt, "the model is never told it can look one up"


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

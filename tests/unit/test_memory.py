"""Tests for the m09a memory store and its two-stage recall.

The bar for every test here is that a store returning *everything* would fail
it. "A relevant memory comes back" is satisfied by a bug, so the assertions are
about what stays out, what gets superseded, and what gets refused.
"""

from __future__ import annotations

import sqlite3
import struct
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import MemoryConfig
from bevel.core.memory.recall import memory_block, recall
from bevel.core.memory.store import MemoryStore, MemoryType, repo_scope
from bevel.core.tools.base import RiskTier, ToolContext, call_tool
from bevel.core.tools.memory import RecallTool, RememberTool, memory_tools
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import Capabilities, ProviderError, TokenCount
from bevel.providers.mock import MockProvider

#: The two memories every precision test is built from. They share no words.
RELEVANT = "the auth middleware broke on empty tokens"
IRRELEVANT = "the caching layer indexes documents nightly"


class CountingProvider(MockProvider):
    """A mock that counts tokens as words, so a budget can be asserted exactly.

    The test picks the counting rule and production never does. That is the
    point: the budget has to be enforced against whatever the endpoint reports,
    and a block that quietly char-counted would fail here.
    """

    def __init__(self, *, known: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.known = known

    async def count_tokens(self, messages: Sequence[Any]) -> TokenCount:
        if not self.known:
            return TokenCount(None, source="endpoint has no /tokenize route")
        return TokenCount(
            sum(len(message.content.split()) for message in messages), source="word-count"
        )


class BrokenEmbedder:
    """An embedder that advertises ``/embeddings`` and then fails to answer it.

    Advertises it on purpose: the point is that a route which exists but is
    broken must degrade to lexical recall, not that a missing one does.
    """

    model = "boom"
    capabilities = Capabilities(supports_embeddings=True)

    async def embed(self, text: str) -> tuple[float, ...]:  # noqa: ARG002
        raise ProviderError("endpoint has no /embeddings route")


def _embed_caps() -> Capabilities:
    return Capabilities(supports_embeddings=True)


def _unit(*values: float) -> tuple[float, ...]:
    norm = sum(value * value for value in values) ** 0.5
    return tuple(value / norm for value in values)


def _store(tmp_path: Path, **config: Any) -> MemoryStore:
    return MemoryStore(tmp_path / "memories.db", config=MemoryConfig(**config))


def _add(
    store: MemoryStore,
    content: str,
    type_: MemoryType = MemoryType.FACT,
    *,
    scope: str = "global",
    **kw: Any,
) -> int:
    return store.add(content=content, type=type_, scope=scope, **kw).id


def _stamp(days_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _age(store: MemoryStore, memory_id: int, created_at: str) -> None:
    """Backdate a row through a second connection.

    Deliberately not a store method: nothing in production rewrites a
    timestamp, so adding one so a test could would be a fake API.
    """
    connection = sqlite3.connect(store.db_path)
    try:
        connection.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", (created_at, memory_id)
        )
        connection.commit()
    finally:
        connection.close()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=30)


# ------------------------------------------------------------------ the store itself


class TestStoreShape:
    def test_constructing_a_store_creates_no_file(self, tmp_path: Path) -> None:
        """`bevel memories` on a fresh machine must leave nothing behind."""
        store = MemoryStore(tmp_path / "nested" / "memories.db")
        try:
            assert not store.db_path.exists()
        finally:
            store.close()

    def test_the_first_write_creates_the_file_and_the_schema(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT)
            assert store.db_path.exists()
            connection = sqlite3.connect(store.db_path)
            try:
                names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                    )
                }
            finally:
                connection.close()
            assert {"memories", "memories_fts"} <= names
        finally:
            store.close()

    def test_a_second_store_on_the_same_path_reads_the_same_rows(self, tmp_path: Path) -> None:
        """Nothing may live only in the writing connection's memory."""
        path = tmp_path / "memories.db"
        first = MemoryStore(path)
        try:
            _add(first, RELEVANT, MemoryType.DECISION)
        finally:
            first.close()

        second = MemoryStore(path)
        try:
            rows = second.list_memories()
            assert [row.content for row in rows] == [RELEVANT]
            assert rows[0].type is MemoryType.DECISION
            assert rows[0].created_at
            assert rows[0].superseded_by is None
            assert rows[0].scope == "global"
            assert rows[0].has_vector is False
        finally:
            second.close()

    def test_reopening_a_store_is_idempotent(self, tmp_path: Path) -> None:
        """The schema runs on every connect, so it has to be IF NOT EXISTS."""
        path = tmp_path / "memories.db"
        for _ in range(3):
            store = MemoryStore(path)
            try:
                _add(store, RELEVANT)
            finally:
                store.close()
        assert len(MemoryStore(path).list_memories()) == 3

    def test_a_blank_memory_is_refused(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            with pytest.raises(ValueError, match="content"):
                store.add(content="   ", type=MemoryType.FACT, scope="global")
        finally:
            store.close()

    def test_get_returns_none_for_an_unknown_id(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            assert store.get(4242) is None
        finally:
            store.close()

    def test_close_is_safe_to_call_twice(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        _add(store, RELEVANT)
        store.close()
        store.close()


# ------------------------------------------------------------------- precision


class TestPrecision:
    """The tests that separate a search from a ``SELECT *``."""

    async def test_an_irrelevant_memory_is_not_recalled(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT)
            _add(store, IRRELEVANT)
            result = await recall(store, "auth middleware", workspace=tmp_path)
            contents = [hit.memory.content for hit in result.memories]
            assert contents == [RELEVANT], f"the unrelated memory was recalled: {contents}"
        finally:
            store.close()

    async def test_a_query_matching_nothing_returns_nothing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT)
            result = await recall(store, "kubernetes helm chart", workspace=tmp_path)
            assert result.memories == ()
        finally:
            store.close()

    async def test_an_empty_query_returns_nothing(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT)
            result = await recall(store, "   ", workspace=tmp_path)
            assert result.memories == ()
        finally:
            store.close()

    async def test_fts_metacharacters_in_a_query_are_words_not_syntax(self, tmp_path: Path) -> None:
        """A query carrying quotes or parentheses must not raise a syntax error."""
        quoted = 'the auth middleware handles "Bearer" tokens (only)'
        store = _store(tmp_path)
        try:
            _add(store, quoted)
            result = await recall(store, 'auth "Bearer" (only)', workspace=tmp_path)
            assert [hit.memory.content for hit in result.memories] == [quoted]
        finally:
            store.close()

    async def test_a_superseded_memory_is_not_recalled_but_is_still_readable(
        self, tmp_path: Path
    ) -> None:
        """Supersede, never evict: the old row stays auditable."""
        replacement = "the auth middleware was removed entirely"
        store = _store(tmp_path)
        try:
            old = _add(store, RELEVANT, MemoryType.DECISION)
            new = _add(store, replacement, MemoryType.DECISION)
            assert store.supersede(old, by=new) is True

            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert [hit.memory.content for hit in result.memories] == [replacement]

            kept = store.get(old)
            assert kept is not None, "the superseded row was deleted"
            assert kept.content == RELEVANT
            assert kept.superseded_by == new
            successor = store.get(new)
            assert successor is not None
            assert successor.superseded_by is None
        finally:
            store.close()

    def test_superseding_a_superseded_memory_is_refused(self, tmp_path: Path) -> None:
        """A chain is auditable only if it stays a chain."""
        store = _store(tmp_path)
        try:
            a = _add(store, "one")
            b = _add(store, "two")
            c = _add(store, "three")
            assert store.supersede(a, by=b) is True
            with pytest.raises(ValueError, match="already superseded"):
                store.supersede(a, by=c)
        finally:
            store.close()

    def test_supersede_refuses_a_missing_row_or_a_self_link(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            a = _add(store, "one")
            with pytest.raises(ValueError, match="no memory"):
                store.supersede(999, by=a)
            with pytest.raises(ValueError, match="itself"):
                store.supersede(a, by=a)
        finally:
            store.close()

    def test_list_memories_can_show_the_superseded(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            a = _add(store, "one")
            b = _add(store, "two")
            store.supersede(a, by=b)
            assert [row.id for row in store.list_memories()] == [b]
            assert sorted(row.id for row in store.list_memories(include_superseded=True)) == [
                a,
                b,
            ]
        finally:
            store.close()


# --------------------------------------------------------------------- ranking


class TestRanking:
    async def test_a_decision_outranks_a_reference_at_equal_relevance(self, tmp_path: Path) -> None:
        """Identical text, different type: the type must decide the order."""
        store = _store(tmp_path)
        try:
            _add(store, "we pin the runtime dependencies", MemoryType.REFERENCE)
            _add(store, "we pin the runtime dependencies", MemoryType.DECISION)
            result = await recall(store, "pin runtime dependencies", workspace=tmp_path)
            types = [hit.memory.type for hit in result.memories]
            assert types == [MemoryType.DECISION, MemoryType.REFERENCE], types
        finally:
            store.close()

    async def test_feedback_outranks_a_reference_at_equal_relevance(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, "always run ruff before committing", MemoryType.REFERENCE)
            _add(store, "always run ruff before committing", MemoryType.FEEDBACK)
            result = await recall(store, "run ruff before committing", workspace=tmp_path)
            assert result.memories[0].memory.type is MemoryType.FEEDBACK
        finally:
            store.close()

    async def test_a_six_month_old_decision_beats_yesterdays_reference(
        self, tmp_path: Path
    ) -> None:
        """The requirement in one line.

        The reference is the newer row and the more recent one by every
        measure, so a recency-dominant ranker puts it first. Only an
        importance-dominant one survives this.
        """
        store = _store(tmp_path)
        try:
            old = _add(store, "auth middleware broke", MemoryType.DECISION)
            recent = _add(store, "auth middleware broke", MemoryType.REFERENCE)
            _age(store, old, _stamp(180))
            _age(store, recent, _stamp(0))

            result = await recall(store, "auth middleware broke", workspace=tmp_path)
            assert len(result.memories) == 2
            winner = result.memories[0].memory
            assert winner.id == old
            assert winner.type is MemoryType.DECISION
        finally:
            store.close()

    async def test_recency_breaks_a_tie_between_two_equal_memories(self, tmp_path: Path) -> None:
        """Same type, same text: the newer one goes first. Recency is a tiebreaker."""
        store = _store(tmp_path)
        try:
            older = _add(store, "auth middleware broke")
            newer = _add(store, "auth middleware broke")
            _age(store, older, _stamp(400))
            _age(store, newer, _stamp(0))
            result = await recall(store, "auth middleware broke", workspace=tmp_path)
            assert [hit.memory.id for hit in result.memories] == [newer, older]
        finally:
            store.close()

    async def test_limit_caps_the_result_and_keeps_the_best(self, tmp_path: Path) -> None:
        store = _store(tmp_path, limit=1)
        try:
            _add(store, "auth middleware broke", MemoryType.REFERENCE)
            _add(store, "auth middleware broke", MemoryType.DECISION)
            result = await recall(store, "auth middleware broke", workspace=tmp_path)
            assert len(result.memories) == 1
            assert result.memories[0].memory.type is MemoryType.DECISION
        finally:
            store.close()

    async def test_scores_are_ordered_highest_first(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, "auth middleware broke")
            _add(store, "auth middleware broke and the router dropped the request too")
            result = await recall(store, "auth middleware broke", workspace=tmp_path)
            assert len(result.memories) == 2
            scores = [hit.score for hit in result.memories]
            assert scores == sorted(scores, reverse=True)
        finally:
            store.close()


# ---------------------------------------------------------------------- stemming


class TestStemming:
    """Porter is a suffix stripper, not a lemma dictionary. Both halves pinned."""

    async def test_regular_forms_are_unified(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, "the caching layer indexes documents nightly")
            for query in ("document index", "document indexing", "documents indexed"):
                result = await recall(store, query, workspace=tmp_path)
                assert [hit.memory.content for hit in result.memories] == [
                    "the caching layer indexes documents nightly"
                ], query
        finally:
            store.close()

    async def test_a_regular_verb_written_one_way_is_found_the_other(self, tmp_path: Path) -> None:
        """The brief's requirement, in the form Porter can actually satisfy."""
        store = _store(tmp_path)
        try:
            _add(store, "auth middleware breaks on empty tokens")
            result = await recall(store, "auth middleware breaking", workspace=tmp_path)
            assert [hit.memory.content for hit in result.memories] == [
                "auth middleware breaks on empty tokens"
            ]
        finally:
            store.close()

    async def test_irregular_forms_are_NOT_unified(self, tmp_path: Path) -> None:
        """``broke`` and ``broken`` are never the same FTS5 token.

        A documented limitation, not a bug to paper over: Porter strips
        suffixes and has no vocabulary, so it cannot map an irregular verb to
        its stem. The query has to be the one word, because recall ORs the
        query's words -- a multi-word query would match on any other word in
        it and prove nothing about this one.
        """
        store = _store(tmp_path)
        try:
            _add(store, "the build broke on a stale cache")
            written = await recall(store, "broke", workspace=tmp_path)
            assert len(written.memories) == 1, "the baseline itself stopped working"

            missed = await recall(store, "broken", workspace=tmp_path)
            assert missed.memories == (), (
                "porter unexpectedly unified broke/broken -- update this test and "
                "the limitation note in store.py"
            )
        finally:
            store.close()


# ----------------------------------------------------------------------- vectors


class TestVectors:
    def test_vectors_are_stored_pre_normalised(self, tmp_path: Path) -> None:
        """Scoring is a dot product over one norm, which measured 2x faster."""
        store = _store(tmp_path, vectors=True)
        try:
            memory_id = _add(store, RELEVANT, vector=(3.0, 4.0), vector_model="m")
            row = store.vector_row(memory_id)
            assert row is not None
            model, blob = row
            assert model == "m"
            values = struct.unpack("<2f", blob)
            assert values[0] ** 2 + values[1] ** 2 == pytest.approx(1.0, abs=1e-6)
        finally:
            store.close()

    def test_a_vector_written_without_a_model_is_refused(self, tmp_path: Path) -> None:
        """An untagged vector is uncomparable, so it must not be stored silently."""
        store = _store(tmp_path, vectors=True)
        try:
            with pytest.raises(ValueError, match="model"):
                store.add(content=RELEVANT, type=MemoryType.FACT, scope="global", vector=(1.0, 0.0))
        finally:
            store.close()

    async def test_cosine_reranks_candidates_that_lexical_ordering_gets_wrong(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="m", capabilities=_embed_caps())
        provider.embeddings = {"auth": _unit(1.0, 0.0, 0.0)}
        try:
            _add(
                store,
                "auth tokens are validated by the middleware",
                vector=_unit(1.0, 0.0, 0.0),
                vector_model="m",
            )
            _add(
                store,
                "auth is a word in the glossary",
                vector=_unit(0.0, 1.0, 0.0),
                vector_model="m",
            )
            store.embedder = provider
            result = await recall(store, "auth", workspace=tmp_path)
            assert result.mode == "vector"
            assert result.memories[0].memory.content.startswith("auth tokens")
        finally:
            store.close()

    async def test_vectors_from_different_models_are_refused_not_averaged(
        self, tmp_path: Path
    ) -> None:
        """Comparing two embedding spaces produces a confident wrong answer."""
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="alpha", capabilities=_embed_caps())
        provider.embeddings = {"auth middleware": _unit(1.0, 0.0)}
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="alpha")
            _add(store, "auth middleware fixed", vector=_unit(0.0, 1.0), vector_model="beta")
            store.embedder = provider

            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical", "mixed-model vectors were scored anyway"
            assert any("model" in note.lower() for note in result.notes), result.notes
            # Both are still recalled lexically: refusing the re-rank is not refusing memory.
            assert len(result.memories) == 2
        finally:
            store.close()

    async def test_vectors_from_another_model_than_the_query_are_refused(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="alpha", capabilities=_embed_caps())
        provider.embeddings = {"auth middleware": _unit(1.0, 0.0)}
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="beta")
            store.embedder = provider
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert any("beta" in note for note in result.notes), result.notes
        finally:
            store.close()

    async def test_a_candidate_without_a_vector_falls_back_to_lexical(self, tmp_path: Path) -> None:
        """A half-vectored candidate set must not be partly re-ranked."""
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="alpha", capabilities=_embed_caps())
        provider.embeddings = {"auth middleware": _unit(1.0, 0.0)}
        try:
            _add(store, "auth middleware broke")  # no vector at all
            store.embedder = provider
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert len(result.memories) == 1
        finally:
            store.close()

    async def test_vectors_are_not_used_when_the_config_has_not_opted_in(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path, vectors=False)
        provider = MockProvider(model="alpha", capabilities=_embed_caps())
        provider.embeddings = {"auth middleware": _unit(1.0, 0.0)}
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="alpha")
            store.embedder = provider
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert any("opted in" in note for note in result.notes), result.notes
        finally:
            store.close()

    async def test_an_endpoint_without_embeddings_says_so(self, tmp_path: Path) -> None:
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="alpha")  # supports_embeddings defaults to False
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="alpha")
            store.embedder = provider
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert any("embeddings" in note for note in result.notes), result.notes
        finally:
            store.close()

    async def test_a_mismatched_stored_dimension_is_refused_not_padded(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="alpha", capabilities=_embed_caps())
        provider.embeddings = {"auth middleware": _unit(1.0, 0.0, 0.0)}
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="alpha")
            store.embedder = provider
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert any("dimension" in note.lower() for note in result.notes), result.notes
        finally:
            store.close()

    async def test_an_embedding_failure_degrades_to_lexical(self, tmp_path: Path) -> None:
        """A dead /embeddings route must not take recall down with it."""
        store = _store(tmp_path, vectors=True)
        try:
            _add(store, "auth middleware broke", vector=_unit(1.0, 0.0), vector_model="boom")
            store.embedder = BrokenEmbedder()
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert len(result.memories) == 1
            assert any("embeddings" in note for note in result.notes), result.notes
        finally:
            store.close()

    async def test_no_embedder_at_all_is_reported_as_lexical_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path, vectors=True)
        try:
            _add(store, "auth middleware broke")
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert result.mode == "lexical"
            assert any("no embedding" in note for note in result.notes), result.notes
        finally:
            store.close()


# ------------------------------------------------------------------ the injected block


class TestInjection:
    # All three share the word "bevel" so one query recalls all three; the
    # budget, not the search, is what has to drop some of them.
    LONG_DECISION = "bevel pins exactly three runtime dependencies and nothing else at all"
    LONG_FACT = "bevel's release process tags a commit on main and then publishes the wheel"
    LONG_REFERENCE = "the bevel documentation site lives under docs and is assembled by hand today"

    async def test_the_block_is_token_budgeted_and_names_what_it_dropped(
        self, tmp_path: Path
    ) -> None:
        """Over budget must report the omissions, never quietly truncate."""
        store = _store(tmp_path, max_injection_tokens=40)
        try:
            decision = _add(store, self.LONG_DECISION, MemoryType.DECISION)
            fact = _add(store, self.LONG_FACT, MemoryType.FACT)
            reference = _add(store, self.LONG_REFERENCE, MemoryType.REFERENCE)
            block = await memory_block(store, CountingProvider(), query="bevel", workspace=tmp_path)

            assert block.text, "the block came back empty"
            assert len(block.included) + len(block.dropped) == 3
            assert block.dropped, "everything fit in 40 tokens, so the test proved nothing"
            for memory in block.included:
                assert memory.content in block.text
            for memory in block.dropped:
                assert memory.content not in block.text, (
                    f"{memory.content!r} was reported dropped but is still in the block"
                )
                assert f"#{memory.id}" in block.text, f"memory {memory.id} was not named"
            assert any("dropped" in note.lower() for note in block.notes), block.notes
            assert block.tokens is not None
            assert block.tokens <= 40
            assert block.source == "word-count"
            assert {m.id for m in block.included} | {m.id for m in block.dropped} == {
                decision,
                fact,
                reference,
            }
        finally:
            store.close()

    async def test_a_token_blind_endpoint_says_budgeting_was_unavailable(
        self, tmp_path: Path
    ) -> None:
        """No /tokenize means no token count. The block must not pretend otherwise."""
        store = _store(tmp_path, max_injection_tokens=40)
        try:
            _add(store, self.LONG_DECISION, MemoryType.DECISION)
            block = await memory_block(
                store, CountingProvider(known=False), query="bevel", workspace=tmp_path
            )
            assert block.tokens is None
            assert "tokenize" in block.source
            assert self.LONG_DECISION in block.text
        finally:
            store.close()

    async def test_no_memories_produces_no_message(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            block = await memory_block(store, MockProvider(), query="bevel", workspace=tmp_path)
            assert block.text == ""
            assert block.message is None
            assert block.included == ()
        finally:
            store.close()

    async def test_the_block_is_a_user_message_not_a_system_one(self, tmp_path: Path) -> None:
        """Merging it into the system message would make it un-evictable."""
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT)
            block = await memory_block(
                store, CountingProvider(), query="auth middleware", workspace=tmp_path
            )
            assert block.message is not None
            assert block.message.role == "user"
            assert RELEVANT in block.message.content
        finally:
            store.close()


# --------------------------------------------------------------------------- scope


class TestScope:
    async def test_a_memory_from_another_repo_is_not_recalled(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT, scope="repo:other")
            _add(store, RELEVANT, scope="repo:mine")
            result = await recall(
                store, "auth middleware", workspace=tmp_path, scopes=("repo:mine",)
            )
            assert [hit.memory.scope for hit in result.memories] == ["repo:mine"]
        finally:
            store.close()

    async def test_the_global_tier_is_always_searched(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            _add(store, RELEVANT, scope="global")
            result = await recall(store, "auth middleware", workspace=tmp_path)
            assert len(result.memories) == 1
        finally:
            store.close()

    async def test_repo_scope_uses_the_origin_remote(self, tmp_path: Path) -> None:
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "remote", "add", "origin", "https://example.test/Owner/Repo.git")
        assert await repo_scope(tmp_path) == "repo:example.test/owner/repo"

    async def test_repo_scope_normalises_ssh_remotes_and_trailing_slashes(
        self, tmp_path: Path
    ) -> None:
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "remote", "add", "origin", "git@example.test:Owner/Repo/")
        assert await repo_scope(tmp_path) == "repo:example.test/owner/repo"

    async def test_repo_scope_falls_back_to_the_path_without_git(self, tmp_path: Path) -> None:
        """A non-repo directory still gets a stable, distinct scope."""
        here = await repo_scope(tmp_path)
        assert here.startswith("path:")
        assert here == await repo_scope(tmp_path)
        assert here != await repo_scope(tmp_path / "elsewhere")


# -------------------------------------------------------------------------- tools


class TestTools:
    async def test_remember_writes_and_recall_reads(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            remember, recall_tool = memory_tools(store, workspace=tmp_path)
            assert remember.risk_tier is RiskTier.WRITE_OUTSIDE
            assert recall_tool.risk_tier is RiskTier.READ
            assert recall_tool.is_parallel_safe
            assert not remember.is_parallel_safe

            ctx = ToolContext(workspace=tmp_path, memory=store)
            written = await remember.execute(
                {"content": "we use uv, not pip", "type": "decision"}, ctx
            )
            assert written.ok, written.error
            read = await recall_tool.execute({"query": "uv or pip"}, ctx)
            assert read.ok, read.error
            assert "we use uv, not pip" in read.content
        finally:
            store.close()

    async def test_remember_tags_a_vector_with_the_model_that_produced_it(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path, vectors=True)
        provider = MockProvider(model="embed-me", capabilities=_embed_caps())
        provider.embeddings = {"the release is tagged from main": _unit(1.0, 0.0)}
        store.embedder = provider
        try:
            remember, _ = memory_tools(store, workspace=tmp_path)
            result = await remember.execute(
                {"content": "the release is tagged from main", "type": "fact"},
                ToolContext(tmp_path, memory=store),
            )
            assert result.ok, result.error
            rows = store.list_memories()
            assert len(rows) == 1
            row = store.vector_row(rows[0].id)
            assert row is not None
            assert row[0] == "embed-me"
        finally:
            store.close()

    async def test_remember_still_stores_text_when_embedding_fails(self, tmp_path: Path) -> None:
        """Losing the vector must not lose the memory."""
        store = _store(tmp_path, vectors=True)
        store.embedder = BrokenEmbedder()
        try:
            remember, _ = memory_tools(store, workspace=tmp_path)
            result = await remember.execute(
                {"content": "the release is tagged from main", "type": "fact"},
                ToolContext(tmp_path, memory=store),
            )
            assert result.ok, result.error
            assert store.list_memories()[0].has_vector is False
            assert "vector" in result.content.lower()
        finally:
            store.close()

    async def test_recall_reports_that_vectors_are_off(self, tmp_path: Path) -> None:
        """Lexical-only must be said out loud, not presented as full recall."""
        store = _store(tmp_path, vectors=False)
        try:
            _add(store, RELEVANT)
            _, recall_tool = memory_tools(store, workspace=tmp_path)
            result = await recall_tool.execute(
                {"query": "auth middleware"}, ToolContext(tmp_path, memory=store)
            )
            assert result.ok
            assert "lexical" in result.content.lower()
        finally:
            store.close()

    async def test_recall_with_no_hits_says_so_rather_than_returning_empty(
        self, tmp_path: Path
    ) -> None:
        store = _store(tmp_path)
        try:
            _, recall_tool = memory_tools(store, workspace=tmp_path)
            result = await recall_tool.execute(
                {"query": "nothing matches this at all"}, ToolContext(tmp_path, memory=store)
            )
            assert result.ok
            assert "no memory" in result.content.lower()
        finally:
            store.close()

    async def test_the_registry_instances_refuse_without_a_store(self) -> None:
        """The registry holds store-less instances; they must fail loudly."""
        ctx = ToolContext(Path("."))
        assert (await RememberTool().execute({"content": "x" * 20, "type": "fact"}, ctx)).failed
        assert (await RecallTool().execute({"query": "anything"}, ctx)).failed

    async def test_an_unknown_type_is_refused_by_the_schema(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        try:
            remember, _ = memory_tools(store, workspace=tmp_path)
            result = await call_tool(
                remember,
                {"content": "x" * 20, "type": "opinion"},
                ToolContext(tmp_path, memory=store),
            )
            assert result.failed
        finally:
            store.close()

    def test_the_two_tools_declare_complete_schemas(self) -> None:
        for tool in (RememberTool(), RecallTool()):
            assert isinstance(tool.spec, ToolSpec)
            assert tool.spec.parameters["additionalProperties"] is False


# ------------------------------------------------------------------ provider seam


class TestEmbeddingSeam:
    async def test_mock_embed_returns_the_scripted_vector(self) -> None:
        provider = MockProvider(model="m", capabilities=_embed_caps())
        provider.embeddings = {"hello": (1.0, 2.0)}
        assert await provider.embed("hello") == (1.0, 2.0)

    async def test_mock_embed_refuses_an_unscripted_text(self) -> None:
        """Silently returning zeros would make every cosine a confident lie."""
        provider = MockProvider(model="m", capabilities=_embed_caps())
        provider.embeddings = {"hello": (1.0, 2.0)}
        with pytest.raises(ProviderError):
            await provider.embed("goodbye")

    async def test_mock_embed_refuses_when_the_endpoint_has_no_embeddings(
        self,
    ) -> None:
        provider = MockProvider(model="m")
        with pytest.raises(ProviderError, match="embeddings"):
            await provider.embed("anything")

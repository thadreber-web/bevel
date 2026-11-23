"""Two-stage recall: FTS5 narrows, cosine re-ranks, tokens budget the injection.

The shape is fixed by measurement, not taste. A full-corpus cosine scan in pure
Python costs about 115 ms at 1 000 memories and 5.4 s at 50 000, which is far
too much to sit inside a live TUI. FTS5 with ``bm25`` narrows to
:data:`~bevel.core.memory.store.CANDIDATE_LIMIT` candidates in single-digit
milliseconds even at 200 000 rows, and re-ranking only those costs about 14 ms.
So: FTS5 first, cosine second, never cosine alone.

Every SQLite call here goes through :func:`asyncio.to_thread` for the same
reason ``Agent._seed`` runs the project-map walk that way: it is CPU-bound work
and the TUI is live.

Ranking is **importance-dominant**. A decision from six months ago must beat
yesterday's trivia, so the type multiplier is the largest term after relevance
and recency is a weak tiebreaker that only separates otherwise-identical
memories.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bevel.core.memory.store import (
    CANDIDATE_LIMIT,
    IMPORTANCE,
    MemoryType,
    cosine,
    query_match,
    repo_scope,
    unpack_vector,
)
from bevel.core.messages import Message
from bevel.providers.base import ProviderError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bevel.core.memory.store import Memory, MemoryStore
    from bevel.providers.base import ModelProvider

__all__ = ["GLOBAL_SCOPE", "MemoryBlock", "RecallHit", "RecallResult", "memory_block", "recall"]

#: How much recency may move a score, at ten years old. Any more and a stale
#: memory would start beating a relevant one.
_MAX_RECENCY_BOOST = 0.02
_SECONDS_PER_YEAR = 365.0 * 24 * 3600
_TEN_YEARS_SECONDS = 10 * _SECONDS_PER_YEAR

#: The user-wide tier, always searched alongside the repository scope.
GLOBAL_SCOPE = "global"


@dataclass(frozen=True)
class RecallHit:
    """One memory that came back, with the score that put it in that position."""

    memory: Memory
    score: float


@dataclass(frozen=True)
class RecallResult:
    """What a recall did, including *how* it did it.

    ``mode`` is ``"vector"`` only when cosine actually re-ranked. Anything that
    degraded to FTS5-only says ``"lexical"`` and explains itself in ``notes``;
    a silent downgrade would be the bad kind of fake.
    """

    memories: tuple[RecallHit, ...] = ()
    mode: str = "lexical"
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MemoryBlock:
    """The recall block, already token-budgeted and ready to inject.

    ``message`` is a standalone ``user`` message. Memory is never merged into
    the system prompt, and it must not be ``messages[0]``, or compaction could
    never evict it.
    """

    text: str
    message: Message | None
    included: tuple[Memory, ...] = ()
    dropped: tuple[Memory, ...] = ()
    tokens: int | None = 0
    source: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


def _recency_factor(now: float, created_at: str) -> float:
    """A tiebreaker, deliberately almost nothing: at most +2%, for a brand-new row.

    A timestamp that will not parse is treated as maximally old rather than as
    "now", so a corrupt row loses a tie instead of winning one.
    """
    try:
        age = now - datetime.fromisoformat(created_at).timestamp()
    except ValueError:
        age = _TEN_YEARS_SECONDS
    years = min(max(age, 0.0), _TEN_YEARS_SECONDS) / _SECONDS_PER_YEAR
    return 1.0 + _MAX_RECENCY_BOOST * (1.0 - years / 10.0)


def _relevance(bm25: float, memory: Memory, similarity: dict[int, float] | None) -> float:
    if similarity is None:
        # bm25 is negated by the store so that higher is better; a row that
        # matched scores above zero, and nothing in the list scores below it.
        return max(bm25, 0.0)
    return similarity[memory.id]


def _rank(
    rows: Sequence[tuple[Memory, float]],
    now: float,
    similarity: dict[int, float] | None = None,
) -> list[RecallHit]:
    """Sort by score descending, id breaking ties so the order is reproducible."""
    hits = [
        RecallHit(
            memory=memory,
            score=_relevance(bm25, memory, similarity)
            * IMPORTANCE[MemoryType(memory.type)]
            * _recency_factor(now, memory.created_at),
        )
        for memory, bm25 in rows
    ]
    hits.sort(key=lambda hit: (-hit.score, hit.memory.id))
    return hits


def _fetch(
    store: MemoryStore, query: str, scopes: Sequence[str]
) -> tuple[list[tuple[Memory, float]], list[str]]:
    """Stage one: the only pass over the whole corpus."""
    match = query_match(query)
    if not match:
        return [], ["recall skipped: the query has no words to match"]
    return list(store.candidates(match, tuple(scopes), CANDIDATE_LIMIT)), []


def _candidate_vectors(
    store: MemoryStore, rows: Sequence[tuple[Memory, float]]
) -> tuple[tuple[str, dict[int, tuple[float, ...]]] | None, str | None]:
    """Load the candidates' vectors, or explain why the vector stage cannot run.

    Comparing vectors from two different embedding models would be comparing
    incompatible spaces, and averaging them would produce a number that means
    nothing. Refusing is the only honest option -- and the memories themselves
    still come back lexically, which is the point.
    """
    vectors: dict[int, tuple[float, ...]] = {}
    models: set[str] = set()
    for memory, _ in rows:
        row = store.vector_row(memory.id)
        if row is None:
            return None, "vector recall skipped: a candidate memory has no embedding"
        models.add(row[0])
        vectors[memory.id] = unpack_vector(row[1])
    if len(models) > 1:
        return None, (
            "vector recall refused: candidate memories were embedded by different models ("
            f"{', '.join(sorted(models))}); those spaces are not comparable"
        )
    if not models:  # pragma: no cover - rows always carry at least one memory
        return None, "vector recall skipped: no candidate memory has an embedding"
    return (next(iter(models)), vectors), None


async def _query_vector(
    embedder: ModelProvider,
    stored_model: str,
    query: str,
    dimension: int,
    notes: list[str],
) -> tuple[float, ...] | None:
    """Embed the query, refusing rather than comparing incomparable numbers."""
    if embedder.model != stored_model:
        notes.append(
            f"vector recall refused: the embedding model is {embedder.model!r} but the stored "
            f"vectors were produced by {stored_model!r}; those spaces are not comparable"
        )
        return None
    if not embedder.capabilities.supports_embeddings:
        notes.append("vector recall is off: the endpoint has no /embeddings route")
        return None
    try:
        vector = await embedder.embed(query)
    except ProviderError as exc:
        notes.append(f"vector recall failed: the /embeddings endpoint returned an error ({exc})")
        return None
    if len(vector) != dimension:
        notes.append(
            f"vector recall refused: the query embedded to {len(vector)} dimensions but the "
            f"stored vectors have {dimension}"
        )
        return None
    return vector


async def recall(
    store: MemoryStore,
    query: str,
    *,
    workspace: str | Path | None = None,
    scopes: Sequence[str] | None = None,
) -> RecallResult:
    """Find the memories that answer ``query`` within ``scopes``.

    Pass ``scopes`` to decide the tiers yourself (the CLI and the tests do);
    otherwise the repository scope is derived from ``workspace`` and the global
    tier is always included.
    """
    if scopes is None:
        root = Path(workspace) if workspace is not None else Path.cwd()
        scopes = [await repo_scope(root), GLOBAL_SCOPE]

    rows, notes = await asyncio.to_thread(_fetch, store, query, scopes)
    if not rows:
        return RecallResult((), "lexical", tuple(notes))

    limit = max(store.config.limit, 1)
    now = time.time()
    similarity: dict[int, float] | None = None

    if not store.config.vectors:
        notes.append("vector recall is off: memory has not opted in to embeddings")
    elif store.embedder is None:
        notes.append("vector recall is off: no embedding model is configured")
    else:
        loaded, note = await asyncio.to_thread(_candidate_vectors, store, rows)
        if note is not None:
            notes.append(note)
        elif loaded is not None:
            model, vectors = loaded
            query_vector = await _query_vector(
                store.embedder, model, query, len(next(iter(vectors.values()))), notes
            )
            if query_vector is not None:
                similarity = {
                    memory_id: cosine(query_vector, vector) for memory_id, vector in vectors.items()
                }

    hits = await asyncio.to_thread(_rank, rows, now, similarity)
    return RecallResult(tuple(hits[:limit]), "vector" if similarity else "lexical", tuple(notes))


_HEADER = "Relevant long-term memories (bevel recall, {mode}):"


def _line(memory: Memory) -> str:
    return f"- [{memory.type}] {memory.content}"


async def memory_block(
    store: MemoryStore,
    provider: ModelProvider,
    *,
    query: str,
    workspace: str | Path | None = None,
) -> MemoryBlock:
    """Build the injection block, trimmed to ``MemoryConfig.max_injection_tokens``.

    Counting goes through ``provider.count_tokens`` -- never a chars/4 guess,
    which this repo does not have and will not add. When the endpoint has no
    tokenizer, ``source`` says so and ``tokens`` stays ``None`` rather than
    reporting an estimate as a fact. When the block cannot fit, it names the
    memories it dropped instead of quietly truncating.
    """
    result = await recall(store, query, workspace=workspace)
    if not result.memories:
        return MemoryBlock("", None, notes=result.notes)

    budget = max(store.config.max_injection_tokens, 0)
    text = _HEADER.format(mode=result.mode)
    notes = list(result.notes)
    included: list[Memory] = []
    dropped: list[Memory] = []
    tokens: int | None = 0
    source = ""

    for hit in result.memories:
        candidate = f"{text}\n{_line(hit.memory)}"
        count = await provider.count_tokens([Message.user(candidate)])
        source = count.source
        # No tokenizer means no honest budget check, so the block is admitted
        # on the endpoint's silence and says which fallback it used.
        used = len(candidate.split()) if count.tokens is None else count.tokens
        if count.tokens is not None and used > budget:
            dropped = [h.memory for h in result.memories[len(included) :]]
            break
        text = candidate
        tokens = count.tokens
        included.append(hit.memory)

    if not included:
        return MemoryBlock("", None, (), tuple(dropped), None, source, tuple(notes))
    if dropped:
        names = ", ".join(f"#{memory.id}" for memory in dropped)
        text = f"{text}\n({len(dropped)} more recalled memories did not fit the {budget} token budget: {names})"
        notes.append(
            f"memory block hit the {budget} token budget: dropped {len(dropped)} of "
            f"{len(result.memories)} recalled memories ({names})"
        )
    return MemoryBlock(
        text,
        Message(role="user", content=text),
        tuple(included),
        tuple(dropped),
        tokens,
        source,
        tuple(notes),
    )

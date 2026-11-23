"""SQLite memory store: typed memories, FTS5 index, supersede chains.

Stdlib ``sqlite3`` only, and the shape deliberately mirrors
:mod:`bevel.core.session.store`: one connection owned by the store, opened on
first write, so importing bevel or constructing an agent never creates a file.

Two tables. ``memories`` holds the rows and everything that filters them --
type, scope, recency, and the ``superseded_by`` link. ``memories_fts`` is a
contentless FTS5 index over the text, sharing rowids with ``memories``. The
split matters: a contentless table cannot be ``DELETE``d from, so superseding
*cannot* be implemented by removing the index row. It is a flag on the real
table, applied in the join, and the stale index row simply never survives the
filter. Nothing is ever deleted.

One limitation to know before trusting a recall. The index is tokenised with
``porter unicode61``, and Porter is a suffix stripper with no vocabulary. It
unifies ``index``/``indexes``/``indexed`` and ``break``/``breaks``/``breaking``,
but it does **not** unify ``broke`` or ``broken`` with ``break``, and it does
not relate ``decided`` to ``decision``. A memory written "auth middleware broke"
is not found by a query for "auth middleware broken". That is the documented
behaviour of the algorithm, not a bug to work around, and
``tests/unit/test_memory.py`` pins both halves of it.
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bevel.config.schema import MemoryConfig
from bevel.core._process import TreeKiller, close_process, uses_new_session

if TYPE_CHECKING:
    from bevel.providers.base import ModelProvider

#: How long the ``git config`` probe behind :func:`repo_scope` may take.
_GIT_TIMEOUT_SECONDS = 10.0

#: Candidates the FTS5 stage hands to the cosine re-rank. Re-ranking more than
#: this is where brute-force cosine stops being cheap.
CANDIDATE_LIMIT = 100

#: Words kept from a query. Underscores split, so ``max_results`` finds
#: ``max results``.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    content       TEXT NOT NULL,
    type          TEXT NOT NULL,
    scope         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    superseded_by INTEGER REFERENCES memories(id),
    vector        BLOB,
    vector_model  TEXT
);
CREATE INDEX IF NOT EXISTS memories_active ON memories (superseded_by, created_at DESC);
CREATE INDEX IF NOT EXISTS memories_scope ON memories (scope, superseded_by);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='',
    tokenize='porter unicode61'
);
"""


class MemoryType(StrEnum):
    """What kind of thing a memory is, which is how much it should be trusted.

    The ranking treats this as the dominant term after relevance: a decision
    and a piece of feedback outrank a reference at equal relevance, because
    "we chose X" and "don't do Y" are worth more to the next turn than "there
    is a page about Y somewhere".
    """

    FACT = "fact"
    FEEDBACK = "feedback"
    DECISION = "decision"
    REFERENCE = "reference"


#: Relative weight of each type in the ranking. Importance-dominant by design.
IMPORTANCE: dict[MemoryType, float] = {
    MemoryType.DECISION: 1.0,
    MemoryType.FEEDBACK: 0.9,
    MemoryType.FACT: 0.7,
    MemoryType.REFERENCE: 0.5,
}


@dataclass(frozen=True)
class Memory:
    """One row of the ``memories`` table."""

    id: int
    content: str
    type: MemoryType
    scope: str
    created_at: str
    superseded_by: int | None
    vector_model: str | None
    has_vector: bool


class MemoryStore:
    """``memories.db``: everything bevel has learned and should not relearn.

    The store owns one connection and does no retrieval itself beyond the raw
    candidate fetch; ranking lives in :mod:`bevel.core.memory.recall`.
    """

    def __init__(self, db_path: Path, *, config: MemoryConfig | None = None) -> None:
        self._db_path = Path(db_path)
        self._config: MemoryConfig = config if config is not None else MemoryConfig()
        self._connection: sqlite3.Connection | None = None
        # Recall is parallel-safe and runs its SQLite work on worker threads,
        # so two agent iterations can be inside this store at once. The
        # connection is opened with check_same_thread=False and every statement
        # goes through _tx(), which holds this lock across a whole statement
        # group -- otherwise two concurrent `add` calls could commit each
        # other's half-finished work.
        self._lock = threading.Lock()
        #: Set by the agent/TUI when an endpoint is available. ``None`` means
        #: memory is lexical-only, and recall says so rather than implying more.
        self.embedder: ModelProvider | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def config(self) -> MemoryConfig:
        return self._config

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        return self._connection

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One serialised unit of work on the shared connection, committed on exit.

        Rolls back on an exception so a refusal partway through cannot leave a
        half-written statement for the next caller's commit to publish.
        """
        with self._lock:
            connection = self._connect()
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    # -- writes -----------------------------------------------------------

    def add(
        self,
        *,
        content: str,
        type: MemoryType,
        scope: str,
        vector: tuple[float, ...] | None = None,
        vector_model: str | None = None,
    ) -> Memory:
        """Store one memory. Never overwrites: correcting the past is ``supersede``.

        A vector must arrive with the model that produced it. An untagged
        vector is not a half-finished row, it is an uncomparable one, so this
        refuses rather than storing something recall would later have to guess
        about.
        """
        text = content.strip()
        if not text:
            raise ValueError("memory content cannot be empty")
        if vector is not None and not vector_model:
            raise ValueError("a stored vector must name the model that produced it")

        now = datetime.now(UTC).isoformat(timespec="seconds")
        blob = pack_vector(vector) if vector is not None else None
        with self._tx() as connection:
            cursor = connection.execute(
                "INSERT INTO memories (content, type, scope, created_at, vector, vector_model)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (text, str(type), scope, now, blob, vector_model),
            )
            row_id = cursor.lastrowid
            assert row_id is not None  # an INSERT always assigns a rowid
            # Contentless FTS5 takes the rowid explicitly; it shares the parent's.
            connection.execute(
                "INSERT INTO memories_fts (rowid, content) VALUES (?, ?)", (row_id, text)
            )
        return Memory(
            id=row_id,
            content=text,
            type=type,
            scope=scope,
            created_at=now,
            superseded_by=None,
            vector_model=vector_model,
            has_vector=vector is not None,
        )

    def supersede(self, memory_id: int, *, by: int) -> bool:
        """Mark ``memory_id`` replaced by ``by``. The old row stays.

        Refuses a second supersede of an already-superseded row: a chain that
        can branch is no longer auditable, and the whole point of keeping the
        row is being able to read the history back.
        """
        if memory_id == by:
            raise ValueError(f"memory {memory_id} cannot supersede itself")
        with self._tx() as connection:
            current = connection.execute(
                "SELECT superseded_by FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"no memory with id {memory_id}")
            if current[0] is not None:
                raise ValueError(
                    f"memory {memory_id} is already superseded by {current[0]};"
                    " supersede the successor instead"
                )
            if connection.execute("SELECT 1 FROM memories WHERE id = ?", (by,)).fetchone() is None:
                raise ValueError(f"no memory with id {by}")
            connection.execute(
                "UPDATE memories SET superseded_by = ? WHERE id = ?", (by, memory_id)
            )
        return True

    # -- reads ------------------------------------------------------------

    def get(self, memory_id: int) -> Memory | None:
        """One row by id, superseded or not. The CLI reads superseded rows this way."""
        with self._tx() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM memories WHERE id = ?",  # noqa: S608 - constant column list
                (memory_id,),
            ).fetchone()
        return None if row is None else _row_to_memory(row)

    def list_memories(self, *, include_superseded: bool = False) -> tuple[Memory, ...]:
        """Every row, newest first. Superseded ones are hidden unless asked for."""
        where = "" if include_superseded else " WHERE superseded_by IS NULL"
        with self._tx() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM memories{where} ORDER BY id DESC",  # noqa: S608 - constant
                (),
            ).fetchall()
        return tuple(_row_to_memory(row) for row in rows)

    def vector_row(self, memory_id: int) -> tuple[str, bytes] | None:
        """The ``(model, blob)`` pair for a row's vector, or ``None``.

        The model comes back with the bytes on purpose: every caller has to
        check them against the model that produced the query embedding before
        the numbers mean anything.
        """
        with self._tx() as connection:
            row = connection.execute(
                "SELECT vector_model, vector FROM memories WHERE id = ? AND vector IS NOT NULL",
                (memory_id,),
            ).fetchone()
        return None if row is None else (str(row[0]), bytes(row[1]))

    def candidates(
        self, match: str, scopes: tuple[str, ...], limit: int = CANDIDATE_LIMIT
    ) -> tuple[tuple[Memory, float], ...]:
        """FTS5 ``MATCH`` narrowed by ``bm25``, with superseded rows already dropped.

        This is stage one of two. It returns at most ``limit`` rows and the
        cosine re-rank only ever sees these, which is the difference between a
        millisecond and a five-second scan.
        """
        if not match:
            return ()
        placeholders = ", ".join("?" for _ in scopes)
        with self._tx() as connection:
            rows = connection.execute(
                # Every interpolated piece is either a module constant or a
                # generated run of "?" placeholders; the values are all bound.
                f"SELECT {_FTS_COLUMNS} FROM memories_fts f"  # noqa: S608
                " JOIN memories m ON m.id = f.rowid"
                " WHERE memories_fts MATCH ? AND m.superseded_by IS NULL"
                f" AND m.scope IN ({placeholders})"
                " ORDER BY bm25(memories_fts) LIMIT ?",
                (match, *scopes, limit),
            ).fetchall()
        # bm25 is negative-is-better; negate so every caller can sort descending.
        return tuple((_row_to_memory(row), -float(row[-1])) for row in rows)


_COLUMNS = "id, content, type, scope, created_at, superseded_by, vector_model, (vector IS NOT NULL)"
_FTS_COLUMNS = (
    "m.id, m.content, m.type, m.scope, m.created_at, m.superseded_by, m.vector_model,"
    " (m.vector IS NOT NULL), bm25(memories_fts)"
)


def _row_to_memory(row: tuple[object, ...]) -> Memory:
    return Memory(
        id=int(str(row[0])),
        content=str(row[1]),
        type=MemoryType(str(row[2])),
        scope=str(row[3]),
        created_at=str(row[4]),
        superseded_by=None if row[5] is None else int(str(row[5])),
        vector_model=None if row[6] is None else str(row[6]),
        has_vector=bool(row[7]),
    )


# --------------------------------------------------------------------- vectors


def pack_vector(vector: tuple[float, ...]) -> bytes:
    """Little-endian float32, pre-normalised to unit length.

    Normalising on the way in is what makes scoring a single dot product
    instead of a dot product and two norms, which measured 2x on the shipping
    interpreter. A zero vector is left alone rather than divided by zero.
    """
    if not vector:
        raise ValueError("a memory vector cannot be empty")
    norm = sum(value * value for value in vector) ** 0.5
    values = vector if norm == 0.0 else tuple(value / norm for value in vector)
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Dot product of two pre-normalised vectors.

    No norm is computed here. Both operands come from :func:`pack_vector` or
    from the same normalisation on the way in, and silently re-normalising
    would hide a mismatch rather than report it.
    """
    return sum(a * b for a, b in zip(left, right, strict=False))


# ---------------------------------------------------------------------- scope


async def repo_scope(workspace: Path) -> str:
    """The memory scope for a workspace: its remote, or its path.

    Per-repository plus a ``global`` tier is the whole model. Deriving it from
    ``remote.origin.url`` rather than the directory name means a second clone
    of the same project, or a checkout in a different folder, still finds the
    decisions made about it.
    """
    url = await _origin_url(workspace)
    if url is None:
        # ``resolve`` touches the filesystem, so it goes off the loop like the
        # git probe above. The scope has to survive the same directory being
        # named relatively or through a link, or one repo gets two memories.
        absolute = await asyncio.to_thread(
            lambda: Path(workspace).expanduser().resolve(strict=False)
        )
        return f"path:{os.path.normcase(str(absolute))}"
    return f"repo:{_normalise_remote(url)}"


async def _origin_url(workspace: Path) -> str | None:
    """``git config --get remote.origin.url``, or ``None``.

    Never raises: no git, no repository, and no origin all mean "fall back to
    the path", not "memory is broken". Runs as a real async subprocess rather
    than a blocking ``subprocess.run``, because recall is called while the TUI
    is live and the loop must never wait on a child process.
    """
    argv = ["git", "-C", str(workspace), "config", "--get", "remote.origin.url"]
    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.DEVNULL,
    }
    if uses_new_session():
        kwargs["start_new_session"] = True
    try:
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
    except (OSError, ValueError, RuntimeError):
        return None

    killer = TreeKiller(proc.pid)
    try:
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=_GIT_TIMEOUT_SECONDS)
        except TimeoutError:
            killer.kill()
            return None
    except asyncio.CancelledError:
        killer.kill()
        raise
    finally:
        killer.close()
        close_process(proc)
    if proc.returncode != 0:
        return None
    url = raw.decode("utf-8", "replace").strip()
    return url or None


def _normalise_remote(url: str) -> str:
    """``git@github.com:Owner/Repo.git`` and ``https://github.com/Owner/Repo``
    both become ``github.com/owner/repo``.

    Lowercased and ``.git``-stripped so that the same project pushed over SSH
    in one checkout and HTTPS in another is one scope, not two.
    """
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    if "://" in cleaned:
        cleaned = cleaned.split("://", 1)[1]
    elif ":" in cleaned:
        # scp-style: `git@host:path`, with no scheme to split on.
        host, _, path = cleaned.partition(":")
        cleaned = f"{host}/{path}"
    cleaned = cleaned.split("@", 1)[-1] if cleaned.startswith(("git@", "ssh://")) else cleaned
    parts = [part for part in cleaned.split("/") if part]
    return "/".join(part.lower() for part in parts)


def query_match(query: str) -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Each word becomes a quoted phrase joined by ``OR``. Quoting is not
    cosmetic: an unquoted ``auth (x OR y)`` or a stray ``"`` is FTS5 syntax,
    and a model that asks about ``"Bearer" tokens`` would otherwise get a
    syntax error where it expected a search.
    """
    words: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(query):
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        words.append(f'"{word.replace(chr(34), chr(34) * 2)}"')
    return " OR ".join(words)


__all__ = [
    "IMPORTANCE",
    "Memory",
    "MemoryStore",
    "MemoryType",
    "cosine",
    "pack_vector",
    "query_match",
    "repo_scope",
    "unpack_vector",
]

"""SQLite session store: turns, tool calls, token and cost accounting.

Stdlib ``sqlite3`` only. A session has to survive the process that created it,
which means every write commits: a turn that completed is in the file before
the loop moves on, so closing the TUI (or losing it to a crash) costs at most
the turn in flight.

The connection opens on first write, so importing bevel or constructing an
agent never creates a database file.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING

from bevel.core.messages import Message, ToolCall

if TYPE_CHECKING:
    from collections.abc import Sequence

    # Type-only: agent.py owns the store, so importing it at runtime would close
    # a cycle. Nothing below needs the class object, only its attributes.
    from bevel.core.agent import ToolExecution

#: Length of the session id. Long enough to type a few characters and still be
#: unique across a machine's history, short enough to be usable on a CLI.
ID_CHARS = 16

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    workspace         TEXT NOT NULL,
    workspace_key     TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    last_activity_at  TEXT NOT NULL,
    turns             INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sessions_activity ON sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS sessions_workspace ON sessions (workspace_key, started_at DESC);

CREATE TABLE IF NOT EXISTS turns (
    id           INTEGER PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    iteration    INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    reasoning    TEXT,
    tool_calls   TEXT,
    tool_call_id TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS turns_session ON turns (session_id, id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id     INTEGER NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    call_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    observation TEXT NOT NULL,
    reason      TEXT NOT NULL,
    fixes       TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tool_calls_turn ON tool_calls (turn_id, id);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    iteration         INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    reasoning_tokens  INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_session ON usage (session_id, id);

CREATE TABLE IF NOT EXISTS todos (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    content     TEXT NOT NULL,
    status      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS todos_session ON todos (session_id, position);
"""


@dataclass(frozen=True)
class SessionInfo:
    """One row of the sessions table, as ``bevel sessions`` shows it."""

    id: str
    workspace: str
    provider: str
    model: str
    started_at: str
    last_activity_at: str
    turns: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class StoredToolCall:
    """A tool call as it happened: what was asked, and what came back."""

    call_id: str
    name: str
    arguments: str
    outcome: str
    observation: str
    reason: str
    fixes: tuple[str, ...]


@dataclass(frozen=True)
class StoredTurn:
    """One message, with the tool calls belonging to it attached.

    ``tool_calls`` is what the model asked for, in the order it asked, filled in
    with what actually happened wherever an execution row exists for that call.
    A call with no execution row keeps an empty ``outcome`` and ``observation``,
    which is how a reader sees that it was asked for and never ran.
    """

    id: int
    iteration: int
    role: str
    content: str
    reasoning: str | None
    tool_calls: tuple[StoredToolCall, ...]
    tool_call_id: str | None


@dataclass(frozen=True)
class UsageRow:
    """Server-reported usage for one iteration."""

    iteration: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int


class TodoStatus(StrEnum):
    """Where a task is. Three states, because more invite model invention."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass(frozen=True)
class Todo:
    """One row of the ``todos`` table.

    Position is not a field: it is the order the rows come back in, which is
    the order the model sent them. A model that wants to reorder a list sends
    it reordered.
    """

    content: str
    status: TodoStatus


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _workspace_key(workspace: Path) -> str:
    """Canonical form used to match a workspace the user typed to a stored one.

    Windows paths arrive however the shell happened to spell them, so matching
    ignores case and separators; ``resolve`` folds the rest.
    """
    resolved = Path(workspace).expanduser().resolve(strict=False)
    return os.path.normcase(str(resolved))


class SessionStore:
    """``sessions.db``: everything a closed session has to leave behind."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._db_path)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # -- sessions ---------------------------------------------------------

    def create_session(self, *, workspace: Path, provider: str, model: str) -> SessionInfo:
        session_id = token_hex(ID_CHARS // 2)
        now = _now()
        resolved = str(Path(workspace).expanduser().resolve(strict=False))
        connection = self._connect()
        connection.execute(
            "INSERT INTO sessions (id, workspace, workspace_key, provider, model,"
            " started_at, last_activity_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, resolved, _workspace_key(workspace), provider, model, now, now),
        )
        connection.commit()
        return SessionInfo(
            id=session_id,
            workspace=resolved,
            provider=provider,
            model=model,
            started_at=now,
            last_activity_at=now,
            turns=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def list_sessions(self) -> tuple[SessionInfo, ...]:
        rows = (
            self._connect()
            .execute(
                "SELECT id, workspace, provider, model, started_at, last_activity_at, turns,"
                " prompt_tokens, completion_tokens, total_tokens"
                " FROM sessions ORDER BY started_at DESC, rowid DESC"
            )
            .fetchall()
        )
        return tuple(_row_to_session(row) for row in rows)

    def get(self, session_id: str) -> SessionInfo | None:
        row = (
            self._connect()
            .execute(
                "SELECT id, workspace, provider, model, started_at, last_activity_at, turns,"
                " prompt_tokens, completion_tokens, total_tokens FROM sessions WHERE id = ?",
                (session_id,),
            )
            .fetchone()
        )
        return None if row is None else _row_to_session(row)

    def resolve(self, prefix: str) -> SessionInfo:
        """The session whose id starts with ``prefix``.

        Refuses an ambiguous prefix rather than guessing which session the user
        meant -- resuming the wrong one silently rewrites an unrelated history.
        """
        rows = (
            self._connect()
            .execute(
                "SELECT id, workspace, provider, model, started_at, last_activity_at, turns,"
                " prompt_tokens, completion_tokens, total_tokens"
                " FROM sessions WHERE id LIKE ? ORDER BY started_at DESC, rowid DESC",
                (f"{prefix}%",),
            )
            .fetchall()
        )
        if not rows:
            raise ValueError(f"no session id starts with {prefix!r}")
        if len(rows) > 1:
            ids = ", ".join(row[0] for row in rows)
            raise ValueError(f"session id {prefix!r} is ambiguous (matches: {ids})")
        return _row_to_session(rows[0])

    def latest_session_id(self, workspace: Path) -> str | None:
        """The most recently started session for ``workspace``, if any."""
        row = (
            self._connect()
            .execute(
                "SELECT id FROM sessions WHERE workspace_key = ?"
                " ORDER BY started_at DESC, rowid DESC LIMIT 1",
                (_workspace_key(workspace),),
            )
            .fetchone()
        )
        return None if row is None else str(row[0])

    # -- transcript -------------------------------------------------------

    def append_turn(self, session_id: str, message: Message, *, iteration: int) -> int:
        """Persist one message. Returns its id, which owns any tool calls."""
        connection = self._connect()
        cursor = connection.execute(
            "INSERT INTO turns (session_id, iteration, role, content, reasoning,"
            " tool_calls, tool_call_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                iteration,
                message.role,
                message.content,
                message.reasoning,
                (
                    json.dumps([_call_to_json(call) for call in message.tool_calls])
                    if message.tool_calls
                    else None
                ),
                message.tool_call_id,
                _now(),
            ),
        )
        connection.execute(
            "UPDATE sessions SET turns = turns + 1, last_activity_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        connection.commit()
        turn_id = cursor.lastrowid
        assert turn_id is not None
        return turn_id

    def append_tool_call(self, session_id: str, turn_id: int, execution: ToolExecution) -> None:
        connection = self._connect()
        connection.execute(
            "INSERT INTO tool_calls (session_id, turn_id, call_id, name, arguments, outcome,"
            " observation, reason, fixes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn_id,
                execution.call.id,
                execution.call.name,
                execution.call.arguments,
                execution.outcome.value,
                execution.result.content if execution.result else "",
                execution.reason,
                json.dumps(list(execution.fixes)),
                _now(),
            ),
        )
        connection.commit()

    def turns(self, session_id: str) -> tuple[StoredTurn, ...]:
        """Every message in a session, oldest first, with its executions."""
        connection = self._connect()
        rows = connection.execute(
            "SELECT id, iteration, role, content, reasoning, tool_calls, tool_call_id"
            " FROM turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        executions: dict[int, list[StoredToolCall]] = {}
        for row in connection.execute(
            "SELECT turn_id, call_id, name, arguments, outcome, observation, reason, fixes"
            " FROM tool_calls WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall():
            executions.setdefault(int(row[0]), []).append(
                StoredToolCall(
                    call_id=str(row[1]),
                    name=str(row[2]),
                    arguments=str(row[3]),
                    outcome=str(row[4]),
                    observation=str(row[5]),
                    reason=str(row[6]),
                    fixes=tuple(json.loads(str(row[7]))),
                )
            )
        turns: list[StoredTurn] = []
        for row in rows:
            turn_id = int(row[0])
            asked = tuple(_call_from_json(item) for item in (json.loads(row[5]) if row[5] else []))
            turns.append(
                StoredTurn(
                    id=turn_id,
                    iteration=int(row[1]),
                    role=str(row[2]),
                    content=str(row[3]),
                    reasoning=None if row[4] is None else str(row[4]),
                    tool_calls=_merge(asked, tuple(executions.get(turn_id, ()))),
                    tool_call_id=None if row[6] is None else str(row[6]),
                )
            )
        return tuple(turns)

    # -- accounting -------------------------------------------------------

    def record_usage(
        self,
        session_id: str,
        *,
        iteration: int,
        prompt: int,
        completion: int,
        reasoning: int,
        total: int,
    ) -> None:
        connection = self._connect()
        connection.execute(
            "INSERT INTO usage (session_id, iteration, prompt_tokens, completion_tokens,"
            " reasoning_tokens, total_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, iteration, prompt, completion, reasoning, total, _now()),
        )
        connection.execute(
            "UPDATE sessions SET prompt_tokens = prompt_tokens + ?, completion_tokens ="
            " completion_tokens + ?, total_tokens = total_tokens + ?, last_activity_at = ?"
            " WHERE id = ?",
            (prompt, completion, total, _now(), session_id),
        )
        connection.commit()

    def usage_rows(self, session_id: str) -> tuple[UsageRow, ...]:
        rows = (
            self._connect()
            .execute(
                "SELECT iteration, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens"
                " FROM usage WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            .fetchall()
        )
        return tuple(
            UsageRow(
                iteration=int(row[0]),
                prompt_tokens=int(row[1]),
                completion_tokens=int(row[2]),
                reasoning_tokens=int(row[3]),
                total_tokens=int(row[4]),
            )
            for row in rows
        )


class TodoStore:
    """``todos`` in ``sessions.db``: the per-session task list the model keeps.

    The list is replaced wholesale rather than edited item by item, so there is
    no id to hand the model and nothing for it to misremember. A model that
    wants to mark a task done sends the whole list again with that task done,
    which is the only interface a small model gets reliably right.

    ``session_id`` is assigned by the agent once a session is open, the same
    way ``MemoryStore.embedder`` is assigned after construction: the tool
    object is built before the session id exists. It is public and mutable on
    purpose; the tool refuses to run while it is ``None``.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None
        self.session_id: str | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._db_path)
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()
        return self._connection

    def _require_session(self) -> str:
        if self.session_id is None:
            raise ValueError("TodoStore has no session: no session has been opened yet")
        return self.session_id

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def replace(self, todos: Sequence[Todo]) -> None:
        """Make the stored list exactly ``todos``, in order.

        Delete and insert share one transaction so a reader never sees the list
        half-replaced, and so a crash leaves the previous list intact.
        """
        session_id = self._require_session()
        connection = self._connect()
        with connection:
            connection.execute("DELETE FROM todos WHERE session_id = ?", (session_id,))
            connection.executemany(
                "INSERT INTO todos (session_id, position, content, status) VALUES (?, ?, ?, ?)",
                [
                    (session_id, position, todo.content, str(todo.status))
                    for position, todo in enumerate(todos)
                ],
            )

    def list_todos(self) -> tuple[Todo, ...]:
        """The list in the order it was sent."""
        session_id = self._require_session()
        rows = (
            self._connect()
            .execute(
                "SELECT content, status FROM todos WHERE session_id = ? ORDER BY position",
                (session_id,),
            )
            .fetchall()
        )
        return tuple(_row_to_todo(row) for row in rows)


def _row_to_todo(row: tuple[object, ...]) -> Todo:
    return Todo(content=str(row[0]), status=TodoStatus(str(row[1])))


def _row_to_session(row: tuple[object, ...]) -> SessionInfo:
    return SessionInfo(
        id=str(row[0]),
        workspace=str(row[1]),
        provider=str(row[2]),
        model=str(row[3]),
        started_at=str(row[4]),
        last_activity_at=str(row[5]),
        turns=int(str(row[6])),
        prompt_tokens=int(str(row[7])),
        completion_tokens=int(str(row[8])),
        total_tokens=int(str(row[9])),
    )


def _call_to_json(call: ToolCall) -> dict[str, str]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _call_from_json(item: dict[str, str]) -> StoredToolCall:
    return StoredToolCall(
        call_id=str(item["id"]),
        name=str(item["name"]),
        arguments=str(item["arguments"]),
        outcome="",
        observation="",
        reason="",
        fixes=(),
    )


def _merge(
    asked: tuple[StoredToolCall, ...], ran: tuple[StoredToolCall, ...]
) -> tuple[StoredToolCall, ...]:
    """The calls as asked for, filled in with what happened to each.

    Keeping the asked-for list as the spine is what lets a reader (and resume)
    see a call that never ran: it is still there, with an empty outcome.
    """
    if not asked:
        return ran
    by_id = {call.call_id: call for call in ran}
    merged: list[StoredToolCall] = []
    for call in asked:
        done = by_id.get(call.call_id)
        merged.append(done if done is not None else call)
    return tuple(merged)


__all__ = [
    "SessionInfo",
    "SessionStore",
    "StoredToolCall",
    "StoredTurn",
    "Todo",
    "TodoStatus",
    "TodoStore",
    "UsageRow",
]

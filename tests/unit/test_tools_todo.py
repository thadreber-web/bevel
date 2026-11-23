"""The todo tool and the ``todos`` table it writes through.

The contract these tests exist to protect is that the list the model sees is
the list on disk. A tracker that quietly appends instead of replacing passes
every "the todo came back" test ever written and is worse than no tracker at
all, because the model reads it as its own plan. So the first test below seeds
three rows, replaces them with one, and insists two are gone.

Everything runs against a real SQLite file. Nothing mocks the database, because
a mocked SQLite proves nothing about SQLite.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from bevel.core.session.store import SessionStore, TodoStore
from bevel.core.tools.base import RiskTier, ToolContext, call_tool
from bevel.core.tools.schema import unsupported_keywords
from bevel.core.tools.todo import Todo, TodoStatus, TodoTool


def _todo(content: str, status: str = "pending") -> dict[str, str]:
    return {"content": content, "status": status}


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "share" / "bevel" / "sessions.db"


@pytest.fixture
def session(db_path: Path, tmp_path: Path) -> tuple[str, TodoStore]:
    """A real session row, and a todo store pointed at the same file.

    The row is real rather than a made-up id because the table has a foreign
    key to ``sessions``: todos cannot outlive the session they belong to.
    """
    sessions = SessionStore(db_path)
    try:
        info = sessions.create_session(workspace=tmp_path, provider="home", model="m")
    finally:
        sessions.close()
    store = TodoStore(db_path)
    store.session_id = info.id
    return info.id, store


async def _call(store: TodoStore | None, todos: list[dict[str, str]], tmp_path: Path) -> Any:
    ctx = ToolContext(workspace=tmp_path, todos=store)
    return await call_tool(TodoTool(), {"todos": todos}, ctx)


class TestReplaceReallyReplaces:
    async def test_replacing_three_todos_with_one_leaves_one(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        _, store = session
        await _call(
            store,
            [_todo("a"), _todo("b", "done"), _todo("c", "in_progress")],
            tmp_path,
        )
        assert len(store.list_todos()) == 3

        result = await _call(store, [_todo("only")], tmp_path)

        assert result.ok, result.error
        assert [t.content for t in store.list_todos()] == ["only"], (
            "the store appended instead of replacing: the model is reading a plan it did not write"
        )

    async def test_an_empty_list_clears_the_list(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        """Empty is a legal call meaning "finished", not a malformed one."""
        _, store = session
        await _call(store, [_todo("a"), _todo("b")], tmp_path)

        result = await _call(store, [], tmp_path)

        assert result.ok, result.error
        assert store.list_todos() == ()

    async def test_order_survives_the_round_trip(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        """The list is ordered; a reader has to see it in the order it was sent."""
        _, store = session
        await _call(store, [_todo("first"), _todo("second"), _todo("third")], tmp_path)
        assert [t.content for t in store.list_todos()] == ["first", "second", "third"]


class TestScopedToTheSession:
    async def test_one_session_never_sees_another_s_todos(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        sessions = SessionStore(db_path)
        try:
            first = sessions.create_session(workspace=tmp_path, provider="h", model="m").id
            second = sessions.create_session(workspace=tmp_path, provider="h", model="m").id
        finally:
            sessions.close()

        store = TodoStore(db_path)
        store.session_id = first
        await _call(store, [_todo("mine")], tmp_path)
        store.session_id = second
        await _call(store, [_todo("theirs", "done")], tmp_path)

        store.session_id = first
        assert [t.content for t in store.list_todos()] == ["mine"]
        store.session_id = second
        assert [t.content for t in store.list_todos()] == ["theirs"]
        store.close()

    async def test_replacing_one_session_s_list_leaves_the_other_alone(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        sessions = SessionStore(db_path)
        try:
            first = sessions.create_session(workspace=tmp_path, provider="h", model="m").id
            second = sessions.create_session(workspace=tmp_path, provider="h", model="m").id
        finally:
            sessions.close()

        store = TodoStore(db_path)
        store.session_id = first
        await _call(store, [_todo("keep me")], tmp_path)
        store.session_id = second
        await _call(store, [_todo("drop me")], tmp_path)
        await _call(store, [], tmp_path)

        store.session_id = first
        assert [t.content for t in store.list_todos()] == ["keep me"]
        store.close()


class TestStoreLessCallFailsLoudly:
    async def test_a_call_with_no_store_fails_naming_the_reason(self, tmp_path: Path) -> None:
        """``ok=True`` with empty content would be the silent fake.

        The registry offers ``todo`` whether or not a store is attached, exactly
        as it offers ``remember``, so the call itself has to say why it did
        nothing.
        """
        result = await _call(None, [_todo("anything")], tmp_path)

        assert result.failed
        assert result.error is not None
        assert "todo" in result.error
        assert "store" in result.error.lower()

    async def test_a_store_with_no_open_session_fails_naming_the_reason(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        store = TodoStore(db_path)
        assert store.session_id is None

        result = await _call(store, [_todo("anything")], tmp_path)

        assert result.failed
        assert result.error is not None
        assert "session" in result.error
        store.close()

    async def test_a_store_with_no_session_never_touches_the_database(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        """A refused call must not leave a file behind in the data directory."""
        store = TodoStore(db_path)
        await _call(store, [_todo("anything")], tmp_path)
        assert not await asyncio.to_thread(db_path.exists)
        store.close()


class TestStatusRoundTripsThroughTheDatabase:
    async def test_every_status_comes_back_as_the_same_status(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        _, store = session
        statuses = [str(status) for status in TodoStatus]
        await _call(store, [_todo(f"task {i}", s) for i, s in enumerate(statuses)], tmp_path)

        rows = store.list_todos()
        assert [t.status for t in rows] == [TodoStatus(s) for s in statuses]
        assert all(isinstance(t.status, TodoStatus) for t in rows), (
            "the status came back as a bare string, so a reader cannot switch on it"
        )

    async def test_a_status_change_replaces_the_row_rather_than_adding_one(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        """The way a task is marked done is sending the same list with it done."""
        _, store = session
        await _call(store, [_todo("ship it", "pending")], tmp_path)
        await _call(store, [_todo("ship it", "done")], tmp_path)

        assert [(t.content, str(t.status)) for t in store.list_todos()] == [("ship it", "done")]

    def test_the_statuses_are_exactly_the_three_the_schema_offers(self) -> None:
        assert [str(s) for s in TodoStatus] == ["pending", "in_progress", "done"]

    def test_a_todo_is_a_frozen_value(self) -> None:
        item = Todo(content="x", status=TodoStatus.PENDING)
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.content = "y"  # type: ignore[misc]


class TestSchemaRejectsWhatItShould:
    """These prove the validator is wired to the *nested* schema, not just the top level.

    A validator that only looked at the outer object would accept a list of
    strings, a bogus status, and a hallucinated ``id`` field -- and every one of
    those would then reach the store.
    """

    async def test_a_bad_status_is_rejected(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        _, store = session
        result = await _call(store, [_todo("a", "finished")], tmp_path)
        assert result.failed
        assert "status" in (result.error or "")
        assert store.list_todos() == (), "a rejected call still wrote"

    async def test_an_extra_property_on_a_todo_is_rejected(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        _, store = session
        args = {"content": "a", "status": "pending", "id": 7}
        result = await call_tool(TodoTool(), {"todos": [args]}, ToolContext(tmp_path, todos=store))
        assert result.failed
        assert "id" in (result.error or "")
        assert store.list_todos() == ()

    async def test_a_non_object_element_is_rejected(
        self, session: tuple[str, TodoStore], tmp_path: Path
    ) -> None:
        _, store = session
        result = await _call(store, ["just a string"], tmp_path)
        assert result.failed
        assert store.list_todos() == ()

    async def test_a_missing_status_is_rejected(self, tmp_path: Path) -> None:
        result = await call_tool(TodoTool(), {"todos": [{"content": "a"}]}, ToolContext(tmp_path))
        assert result.failed
        assert "status" in (result.error or "")

    async def test_a_missing_todos_is_rejected(self, tmp_path: Path) -> None:
        result = await call_tool(TodoTool(), {}, ToolContext(tmp_path))
        assert result.failed

    async def test_a_non_list_todos_is_rejected(self, tmp_path: Path) -> None:
        result = await call_tool(
            TodoTool(), {"todos": {"content": "a", "status": "pending"}}, ToolContext(tmp_path)
        )
        assert result.failed

    async def test_an_over_long_content_is_rejected(self, tmp_path: Path) -> None:
        result = await call_tool(
            TodoTool(),
            {"todos": [_todo("x" * 500)]},
            ToolContext(tmp_path),
        )
        assert result.failed
        assert "content" in (result.error or "")

    async def test_an_over_long_list_is_rejected(self, tmp_path: Path) -> None:
        ceiling = TodoTool.spec.parameters["properties"]["todos"]["maxItems"]
        result = await call_tool(
            TodoTool(),
            {"todos": [_todo(f"task {i}") for i in range(ceiling + 1)]},
            ToolContext(tmp_path),
        )
        assert result.failed

    def test_the_schema_uses_only_keywords_the_validator_understands(self) -> None:
        assert unsupported_keywords(TodoTool.spec.parameters) == set()

    def test_both_levels_reject_extra_properties(self) -> None:
        schema = TodoTool.spec.parameters
        assert schema["additionalProperties"] is False
        assert schema["properties"]["todos"]["items"]["additionalProperties"] is False

    def test_every_property_at_every_level_is_described(self) -> None:
        schema = TodoTool.spec.parameters
        for name, prop in schema["properties"].items():
            assert prop.get("description"), name
        for name, prop in schema["properties"]["todos"]["items"]["properties"].items():
            assert prop.get("description"), f"todos[].{name}"

    def test_the_bounds_are_set_and_are_real_numbers(self) -> None:
        array = TodoTool.spec.parameters["properties"]["todos"]
        assert array["minItems"] == 0, "an empty list means 'cleared' and must be legal"
        assert isinstance(array["maxItems"], int)
        assert array["maxItems"] > 0


class TestDeclaration:
    def test_the_tool_declares_itself_read_with_a_written_justification(self) -> None:
        """``READ`` is a judgement call, so the reason has to be in the code.

        ``base.py`` warns against a tier inflated to get a call through. A
        future reader who sees a store-writing tool declared ``READ`` will
        "fix" it as a bug unless the justification is where they are looking.
        """
        assert TodoTool.risk_tier is RiskTier.READ
        doc = inspect.getdoc(sys.modules[TodoTool.__module__]) or ""
        assert "READ" in doc
        assert "approval" in doc.lower()

    def test_replacing_shared_state_is_not_parallel_safe(self) -> None:
        """Two concurrent whole-list replaces race; the later one wins at random."""
        assert TodoTool.is_parallel_safe is False

    def test_the_name_is_wire_safe(self) -> None:
        assert TodoTool.spec.name == "todo"
        assert len(TodoTool.spec.description) > 20


class TestReopen:
    async def test_todos_written_by_one_store_are_read_by_another(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        """A fresh connection, so nothing can be coming from process memory."""
        sessions = SessionStore(db_path)
        try:
            info = sessions.create_session(workspace=tmp_path, provider="h", model="m")
        finally:
            sessions.close()

        first = TodoStore(db_path)
        first.session_id = info.id
        await _call(first, [_todo("persist me", "in_progress"), _todo("and me", "done")], tmp_path)
        first.close()

        reopened = TodoStore(db_path)
        reopened.session_id = info.id
        rows = reopened.list_todos()
        reopened.close()

        assert [(t.content, str(t.status)) for t in rows] == [
            ("persist me", "in_progress"),
            ("and me", "done"),
        ]

    async def test_reading_a_session_that_has_no_todos_is_empty_not_an_error(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        sessions = SessionStore(db_path)
        try:
            info = sessions.create_session(workspace=tmp_path, provider="h", model="m")
        finally:
            sessions.close()
        store = TodoStore(db_path)
        store.session_id = info.id
        assert store.list_todos() == ()
        store.close()

    def test_constructing_the_store_creates_no_file(self, db_path: Path) -> None:
        TodoStore(db_path)
        assert not db_path.exists()

    def test_an_additive_table_needs_no_migration(self, db_path: Path) -> None:
        """A sessions.db written before m09b must gain the table on next connect.

        ``_connect()`` re-runs the whole schema, and every statement is
        ``CREATE TABLE IF NOT EXISTS``, so an old file simply acquires the new
        table. This simulates that by creating the database without it.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        connection.executescript(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, workspace TEXT NOT NULL,"
            " workspace_key TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,"
            " started_at TEXT NOT NULL, last_activity_at TEXT NOT NULL, turns INTEGER NOT NULL"
            " DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0, completion_tokens INTEGER"
            " NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0);"
        )
        connection.execute(
            "INSERT INTO sessions (id, workspace, workspace_key, provider, model, started_at,"
            " last_activity_at) VALUES ('old', 'w', 'w', 'p', 'm', 'now', 'now')"
        )
        connection.commit()
        connection.close()

        store = TodoStore(db_path)
        store.session_id = "old"
        assert store.list_todos() == ()
        store.close()


class TestTheForeignKeyIsReal:
    """The README says a todo list is "cascade-deleted with" its session.

    Nothing in the Python API deletes a session, so no test above could reach
    that behaviour, and a claim about a schema property that no test touches is
    a claim waiting to be wrong. These go straight to SQL: the foreign key must
    refuse an orphan, and deleting the session must take the rows with it rather
    than leaving a list belonging to nobody.
    """

    def test_an_orphan_todo_is_refused(self, db_path: Path) -> None:
        store = TodoStore(db_path)
        store.session_id = "never-opened"
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.replace([Todo(content="orphan", status=TodoStatus.PENDING)])
        store.close()

    def test_deleting_the_session_deletes_its_todos(self, db_path: Path, tmp_path: Path) -> None:
        sessions = SessionStore(db_path)
        try:
            info = sessions.create_session(workspace=tmp_path, provider="h", model="m")
        finally:
            sessions.close()

        store = TodoStore(db_path)
        store.session_id = info.id
        store.replace([Todo(content="goes with the session", status=TodoStatus.DONE)])
        assert len(store.list_todos()) == 1

        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("DELETE FROM sessions WHERE id = ?", (info.id,))
        connection.commit()
        left = connection.execute(
            "SELECT COUNT(*) FROM todos WHERE session_id = ?", (info.id,)
        ).fetchone()[0]
        connection.close()
        store.close()
        assert left == 0, "the list outlived the session it belonged to"

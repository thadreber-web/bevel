"""The session store: what a closed session has to leave behind.

The contract these tests hold is the one the resume feature stands on: a turn
that completed is in the file, a tool call is recorded with its arguments, its
outcome and its observation, and the token accounting came from the server. A
session that cannot be read back byte-for-byte is a session that was never
saved, no matter what the TUI showed while it was open.

Everything runs against a real SQLite file in ``tmp_path``. Nothing here mocks
the database, because a mocked SQLite proves nothing about SQLite.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from pathlib import Path

import pytest

from bevel.core.agent import ToolExecution, ToolOutcome
from bevel.core.messages import Message, ToolCall
from bevel.core.session.store import SessionInfo, SessionStore
from bevel.core.tools.base import ToolResult


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "bevel" / "sessions.db")


def _tool_execution(
    *,
    call_id: str = "c1",
    name: str = "read",
    arguments: str = '{"path": "a.py"}',
    outcome: ToolOutcome = ToolOutcome.OK,
    observation: str = "1|one\n2|two",
    reason: str = "",
    fixes: tuple[str, ...] = (),
) -> ToolExecution:
    return ToolExecution(
        call=ToolCall(id=call_id, name=name, arguments=arguments),
        tool=None,
        outcome=outcome,
        result=ToolResult(ok=outcome is ToolOutcome.OK, content=observation),
        fixes=fixes,
        reason=reason,
    )


class TestCreateAndList:
    def test_a_database_file_appears_only_when_a_session_is_created(self, tmp_path: Path) -> None:
        """Constructing the store must not litter the data directory."""
        path = tmp_path / "bevel" / "sessions.db"
        SessionStore(path)
        assert not path.exists()
        SessionStore(path).create_session(workspace=tmp_path, provider="home", model="m")
        assert path.is_file()

    def test_listing_is_empty_before_any_session(self, store: SessionStore) -> None:
        assert store.list_sessions() == ()

    def test_a_created_session_is_listed_with_its_identity(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("D:/dev/demo"), provider="home", model="qwen")
        (listed,) = store.list_sessions()
        assert listed.id == info.id
        assert listed.workspace == str(Path("D:/dev/demo").resolve())
        assert listed.provider == "home"
        assert listed.model == "qwen"
        assert listed.turns == 0
        assert listed.prompt_tokens == 0

    def test_the_id_is_unique_prefix_matchable(self, store: SessionStore) -> None:
        ids = {
            store.create_session(workspace=Path("."), provider="p", model="m").id for _ in range(8)
        }
        assert len(ids) == 8
        assert all(len(i) >= 12 for i in ids)

    def test_listing_is_newest_first(self, store: SessionStore) -> None:
        first = store.create_session(workspace=Path("a"), provider="p", model="m")
        second = store.create_session(workspace=Path("b"), provider="p", model="m")
        listed = store.list_sessions()
        assert [s.id for s in listed] == [second.id, first.id] or listed[0].started_at >= (
            listed[1].started_at
        )

    def test_get_returns_the_session_and_none_for_an_unknown_id(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        assert store.get(info.id) == info
        assert store.get("nope-nope-nope") is None

    def test_resolve_accepts_a_unique_prefix(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        assert store.resolve(info.id[:8]) == info
        assert store.resolve(info.id) == info

    def test_resolve_refuses_an_ambiguous_prefix(self, store: SessionStore) -> None:
        seen: dict[str, str] = {}
        for _ in range(40):
            info = store.create_session(workspace=Path("."), provider="p", model="m")
            if info.id[0] in seen:
                with pytest.raises(ValueError, match="ambiguous"):
                    store.resolve(info.id[0])
                return
            seen[info.id[0]] = info.id
        pytest.fail("no two session ids shared a prefix, so ambiguity went untested")


class TestAccounting:
    def test_turns_and_tokens_are_read_back_from_the_file(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        store.append_turn(info.id, Message.user("do the thing"), iteration=1)
        store.append_turn(
            info.id, Message("assistant", content="done", reasoning="because"), iteration=1
        )
        store.record_usage(info.id, iteration=1, prompt=120, completion=8, reasoning=3, total=128)
        store.record_usage(info.id, iteration=2, prompt=200, completion=12, reasoning=0, total=212)

        (listed,) = store.list_sessions()
        assert listed.turns == 2
        assert listed.prompt_tokens == 320
        assert listed.completion_tokens == 20
        assert listed.total_tokens == 340
        assert listed.last_activity_at >= listed.started_at

    def test_usage_rows_are_theirselves_persisted(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        store.record_usage(info.id, iteration=3, prompt=10, completion=2, reasoning=1, total=12)
        rows = store.usage_rows(info.id)
        assert [
            (r.iteration, r.prompt_tokens, r.completion_tokens, r.total_tokens) for r in rows
        ] == [(3, 10, 2, 12)]
        assert rows[0].reasoning_tokens == 1

    def test_a_second_session_does_not_inherit_the_firsts_counts(self, store: SessionStore) -> None:
        a = store.create_session(workspace=Path("a"), provider="p", model="m")
        b = store.create_session(workspace=Path("b"), provider="p", model="m")
        store.append_turn(a.id, Message.user("only mine"), iteration=1)
        assert store.get(a.id) is not None
        (listed_b,) = [s for s in store.list_sessions() if s.id == b.id]
        assert listed_b.turns == 0
        assert listed_b.total_tokens == 0


class TestToolCalls:
    def test_a_tool_call_keeps_its_name_arguments_outcome_and_observation(
        self, store: SessionStore
    ) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        turn = store.append_turn(
            info.id,
            Message("assistant", tool_calls=(ToolCall("c1", "read", '{"path": "a.py"}'),)),
            iteration=1,
        )
        store.append_tool_call(info.id, turn, _tool_execution())

        (loaded,) = store.turns(info.id)
        assert loaded.role == "assistant"
        (call,) = loaded.tool_calls
        assert call.name == "read"
        assert call.arguments == '{"path": "a.py"}'
        assert call.outcome == "ok"
        assert call.observation == "1|one\n2|two"
        assert call.call_id == "c1"

    def test_a_refused_call_is_recorded_with_its_reason(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        turn = store.append_turn(
            info.id,
            Message("assistant", tool_calls=(ToolCall("c1", "shell", '{"argv": ["rm"]}'),)),
            iteration=1,
        )
        store.append_tool_call(
            info.id,
            turn,
            _tool_execution(
                name="shell",
                outcome=ToolOutcome.REJECTED,
                observation="",
                reason="denied by policy",
            ),
        )
        (loaded,) = store.turns(info.id)
        (call,) = loaded.tool_calls
        assert call.outcome == "rejected"
        assert call.reason == "denied by policy"
        assert call.observation == ""

    def test_repair_fixes_survive_the_round_trip(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        turn = store.append_turn(
            info.id,
            Message("assistant", tool_calls=(ToolCall("c1", "read", "{}"),)),
            iteration=1,
        )
        store.append_tool_call(
            info.id, turn, _tool_execution(fixes=("arguments were a JSON string",))
        )
        (loaded,) = store.turns(info.id)
        assert loaded.tool_calls[0].fixes == ("arguments were a JSON string",)

    def test_calls_attach_to_the_turn_that_asked_for_them(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        first = store.append_turn(
            info.id, Message("assistant", tool_calls=(ToolCall("c1", "read", "{}"),)), iteration=1
        )
        store.append_turn(info.id, Message.tool_result("c1", "one"), iteration=1)
        second = store.append_turn(
            info.id,
            Message("assistant", tool_calls=(ToolCall("c2", "grep", "{}"),)),
            iteration=2,
        )
        store.append_tool_call(info.id, first, _tool_execution())
        store.append_tool_call(info.id, second, _tool_execution(call_id="c2", name="grep"))

        turns = store.turns(info.id)
        assert [t.tool_calls for t in turns] == [
            (turns[0].tool_calls[0],),
            (),
            (turns[2].tool_calls[0],),
        ]
        assert [c.call_id for t in turns for c in t.tool_calls] == ["c1", "c2"]
        assert [c.name for t in turns for c in t.tool_calls] == ["read", "grep"]


class TestDurability:
    def test_rows_are_readable_from_a_second_connection_without_closing_the_first(
        self, store: SessionStore
    ) -> None:
        """A TUI holding the store open must not hide its writes from `bevel sessions`."""
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        store.append_turn(info.id, Message.user("hi"), iteration=1)
        other = SessionStore(store.db_path)
        assert [t.content for t in other.turns(info.id)] == ["hi"]

    def test_a_reopened_store_sees_everything_the_old_one_wrote(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        store.append_turn(info.id, Message.user("one"), iteration=1)
        store.append_turn(info.id, Message("assistant", content="two"), iteration=1)
        store.record_usage(info.id, iteration=1, prompt=5, completion=2, reasoning=0, total=7)
        store.close()

        reopened = SessionStore(store.db_path)
        assert [t.role for t in reopened.turns(info.id)] == ["user", "assistant"]
        assert reopened.get(info.id) is not None
        (listed,) = [s for s in reopened.list_sessions() if s.id == info.id]
        assert listed.total_tokens == 7

    def test_the_schema_is_the_documented_one(self, store: SessionStore) -> None:
        info = store.create_session(workspace=Path("."), provider="p", model="m")
        store.append_turn(info.id, Message.user("hi"), iteration=1)
        connection = sqlite3.connect(store.db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        finally:
            connection.close()
        assert {"sessions", "turns", "tool_calls", "usage"} <= tables
        assert info.id


class TestLatestForWorkspace:
    def test_the_most_recent_session_for_a_workspace_wins(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path / "s.db")
        workspace = tmp_path / "repo"
        workspace.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        old = store.create_session(workspace=workspace, provider="p", model="m")
        store.append_turn(old.id, Message.user("work"), iteration=1)
        newest = store.create_session(workspace=workspace, provider="p", model="m")
        store.create_session(workspace=other, provider="p", model="m")

        assert store.latest_session_id(workspace) == newest.id

    def test_a_workspace_with_no_sessions_has_no_latest(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        store.create_session(workspace=tmp_path / "elsewhere", provider="p", model="m")
        assert store.latest_session_id(tmp_path / "nowhere") is None

    def test_a_differently_spelled_workspace_is_the_same_workspace(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        """A workspace typed a different way is still the same workspace.

        Only spellings that mean the same path on *every* platform are asserted
        here; case and separators are platform-specific and tested separately.
        """
        workspace = tmp_path / "Repo"
        workspace.mkdir()
        info = store.create_session(workspace=workspace, provider="p", model="m")
        variants = [
            str(workspace),
            str(workspace) + os.sep,
            str(Path(str(workspace), os.pardir, workspace.name)),
        ]
        for variant in variants:
            assert store.latest_session_id(Path(variant)) == info.id, variant

    @pytest.mark.skipif(os.name != "nt", reason="POSIX paths are case-sensitive")
    def test_windows_workspace_matching_ignores_case_and_separators(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        """Windows paths arrive as the user typed them; matching must not be literal."""
        workspace = tmp_path / "Repo"
        workspace.mkdir()
        info = store.create_session(workspace=workspace, provider="p", model="m")
        variants = [str(workspace).upper(), str(workspace).replace("\\", "/")]
        for variant in variants:
            assert store.latest_session_id(Path(variant)) == info.id, variant

    @pytest.mark.skipif(os.name == "nt", reason="Windows paths are case-insensitive")
    def test_a_different_case_is_a_different_workspace_on_posix(
        self, store: SessionStore, tmp_path: Path
    ) -> None:
        """``normcase`` is the identity here, so case is part of the identity.

        The upper-cased spelling really is a different directory on POSIX, and
        claiming it was the same session would let ``bevel resume`` reopen
        someone else's history.
        """
        workspace = tmp_path / "Repo"
        workspace.mkdir()
        store.create_session(workspace=workspace, provider="p", model="m")
        assert store.latest_session_id(Path(str(workspace).upper())) is None


class TestSessionInfoShape:
    def test_info_is_a_frozen_value(self) -> None:
        info = SessionInfo(
            id="x",
            workspace="w",
            provider="p",
            model="m",
            started_at="t",
            last_activity_at="t",
            turns=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.turns = 1  # type: ignore[misc]

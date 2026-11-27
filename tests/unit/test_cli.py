"""Tests for the CLI contract.

The rules these enforce are the project's honesty rules: nothing may be
accepted and then silently ignored, and unimplemented commands must fail
loudly while naming the milestone that delivers them.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from bevel import __version__
from bevel.cli import _PENDING, build_parser, main
from bevel.config import loader
from bevel.core.checkpoint import snapshot
from bevel.core.memory.store import MemoryStore, MemoryType
from bevel.core.messages import Message
from bevel.core.session import store as session_store
from bevel.core.session.store import SessionStore, Todo, TodoStatus, TodoStore


def _pin_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the session store at a file in tmp_path, as the TUI tests do."""
    path = tmp_path / "share" / "sessions.db"
    monkeypatch.setattr(loader, "session_db_path", lambda: path)
    return path


#: The real stdio MCP server every m10 slice spawns. ``bevel mcp list`` is
#: tested against it rather than a stub, because the command's only job is to
#: report whether a server is genuinely up.
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"


class TestParser:
    def test_no_flag_is_parsed_without_being_read(self) -> None:
        """Every option the parser accepts must be one the current code acts on.

        A flag that parses but is never read lets `bevel --model x` quietly do
        nothing. Flags arrive with the milestone that implements them.
        """
        options = {option for action in build_parser()._actions for option in action.option_strings}
        assert options == {"--version", "-h", "--help"}

    def test_pending_subcommands_are_registered(self) -> None:
        parser = build_parser()
        for name in _PENDING:
            assert parser.parse_args([name]).command == name


class TestVersion:
    def test_version_prints_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0
        assert capsys.readouterr().out.strip() == f"bevel {__version__}"


class TestDoctor:
    def test_doctor_exits_zero_and_reports(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("bevel doctor")
        assert "best tier" in out
        for name in ("landlock", "appcontainer", "job_object", "windows_sandbox"):
            assert name in out

    def test_doctor_labels_its_own_limitation(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Static probing is not verification; the output must say so."""
        main(["doctor"])
        out = capsys.readouterr().out
        assert "Static probing only" in out

    def test_doctor_never_claims_a_self_test_ran(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The self-test is m12. Claiming it verified anything would be a fake."""
        main(["doctor"])
        out = capsys.readouterr().out
        assert "verified at runtime" not in out
        assert "bevel enforces the T0" in out


class TestPendingSubcommands:
    @pytest.mark.parametrize("command", sorted(_PENDING))
    def test_each_pending_command_fails_naming_its_milestone(
        self, command: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([command]) == 2
        err = capsys.readouterr().err
        assert f"bevel {command}: not implemented yet" in err
        assert _PENDING[command] in err

    def test_no_pending_command_is_silently_accepted(self) -> None:
        """A planned command must never return success."""
        assert all(main([command]) != 0 for command in _PENDING)


class TestPr:
    """``bevel pr create`` / ``bevel pr update`` -- the gh-backed PR surface (m11b).

    These spawn ``tests/fixtures/gh.py`` for real, the way ``TestMcp`` spawns the
    MCP fixture: the command's job is to build gh's argv and report what gh
    answered, and a stub would print whatever it was told to. The real gh is
    injected as the program via ``bevel.cli._GH``; bevel's own validator still
    checks the logical argv, and the fixture refuses anything off the allowlist.
    """

    GH_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gh.py"

    def _inject_gh(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str) -> Path:
        """Point the CLI at the fixture and record what it was asked to do."""
        seen = tmp_path / "gh-seen.jsonl"
        monkeypatch.setattr("bevel.cli._GH", (sys.executable, str(self.GH_FIXTURE)))
        monkeypatch.setenv("BEVEL_GH_FIXTURE_SEEN", str(seen))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return seen

    def _repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, pushed: bool = True
    ) -> Path:
        """A real repo on a ``feature`` branch, optionally pushed to a bare origin."""
        empty = tmp_path / "empty-gitconfig"
        empty.write_text("", encoding="utf-8")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        identity = (
            "-c",
            "user.name=bevel",
            "-c",
            "user.email=bevel@localhost",
            "-c",
            "commit.gpgsign=false",
        )

        def git(*args: str) -> None:
            subprocess.run(
                ["git", *identity, *args],
                cwd=root,
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
                timeout=60,
            )

        root = tmp_path / "repo"
        root.mkdir()
        git("init", "-q", "-b", "main")
        (root / "a.txt").write_text("a\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-q", "--no-verify", "-m", "seed")
        if pushed:
            bare = tmp_path / "origin.git"
            bare.mkdir()
            subprocess.run(
                ["git", *identity, "init", "--bare", "-q", "-b", "main"],
                cwd=bare,
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
                timeout=60,
            )
            git("remote", "add", "origin", str(bare))
            git("push", "-q", "-u", "origin", "main")
            git("remote", "set-head", "origin", "-a")
            git("switch", "-q", "-c", "feature")
            (root / "f.txt").write_text("f\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-q", "--no-verify", "-m", "work")
            git("push", "-q", "-u", "origin", "feature")
        monkeypatch.chdir(root)
        return root

    def test_pr_is_no_longer_a_pending_command(self) -> None:
        """The honesty rule: a command that works must not announce itself as unbuilt."""
        assert "pr" not in _PENDING

    def test_the_subcommand_takes_only_positionals(self) -> None:
        """A ``--flag`` here would have to be read by this milestone."""
        subparser = next(
            action.choices["pr"]
            for action in build_parser()._actions
            if action.choices and "pr" in action.choices
        )
        options = {option for action in subparser._actions for option in action.option_strings}
        assert options == {"-h", "--help"}

    def test_no_action_is_a_usage_error_that_prints_the_actions(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["pr"]) == 2
        err = capsys.readouterr().err
        assert "create" in err
        assert "update" in err

    def test_an_unknown_action_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["pr", "merge"])
        assert excinfo.value.code == 2

    def test_create_prints_the_pr_gh_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._repo(tmp_path, monkeypatch)
        self._inject_gh(tmp_path, monkeypatch)
        assert main(["pr", "create"]) == 0
        out = capsys.readouterr().out
        assert "https://github.com/bevel-fixture/repo/pull/42" in out

    def test_create_on_an_unpushed_branch_exits_one_naming_the_push(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._repo(tmp_path, monkeypatch, pushed=False)
        seen = self._inject_gh(tmp_path, monkeypatch)
        assert main(["pr", "create"]) == 1
        err = capsys.readouterr().err
        assert "git push -u origin" in err
        assert not seen.exists(), "the refusal happens before gh is spawned"

    def test_an_unauthenticated_gh_exits_one_naming_the_fix(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._repo(tmp_path, monkeypatch)
        self._inject_gh(tmp_path, monkeypatch, BEVEL_GH_FIXTURE_MODE="unauthenticated")
        assert main(["pr", "create"]) == 1
        err = capsys.readouterr().err
        assert "gh auth login" in err

    def test_update_prints_the_pr_gh_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._repo(tmp_path, monkeypatch)
        self._inject_gh(tmp_path, monkeypatch)
        assert main(["pr", "update", "7", "A new title"]) == 0
        out = capsys.readouterr().out
        assert "#7" in out
        assert "https://github.com/bevel-fixture/repo/pull/7" in out

    def test_update_without_a_number_is_a_usage_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._repo(tmp_path, monkeypatch)
        self._inject_gh(tmp_path, monkeypatch)
        assert main(["pr", "update"]) == 2

    def test_no_token_value_is_ever_printed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Presence-only checks: a token is not print material (same rule as mcp list)."""
        self._repo(tmp_path, monkeypatch)
        self._inject_gh(tmp_path, monkeypatch)
        monkeypatch.setenv("GH_TOKEN", "ghp_hunter2_super_secret")
        assert main(["pr", "create"]) == 0
        captured = capsys.readouterr()
        assert "ghp_hunter2_super_secret" not in captured.out + captured.err


class TestSessions:
    def test_no_sessions_says_so_and_names_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        assert main(["sessions"]) == 0
        out = capsys.readouterr().out
        assert "no sessions stored yet" in out
        assert str(db) in out
        assert not db.exists(), "listing sessions must not create a database"

    def test_a_stored_session_is_listed_with_what_you_need_to_pick_it(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        info = store.create_session(workspace=tmp_path, provider="vllm", model="qwen")
        store.append_turn(info.id, Message.user("do the thing"), iteration=1)
        store.record_usage(
            info.id, iteration=1, prompt=1200, completion=340, reasoning=0, total=1540
        )
        store.close()

        assert main(["sessions"]) == 0
        out = capsys.readouterr().out
        header, row, *_ = out.splitlines()
        assert info.id in row
        assert f"{info.prompt_tokens:,}" in row
        assert f"{info.completion_tokens:,}" in row
        assert str(tmp_path) in row
        # The turn count is a bare column; check it is there rather than guessing
        # which of the numeric columns it is.
        assert " 1 " in f" {row} "
        for column in (
            "id",
            "started",
            "last activity",
            "turns",
            "prompt",
            "completion",
            "workspace",
        ):
            assert column in header
        assert str(db) in out

    def test_the_newest_session_is_listed_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        first = store.create_session(workspace=tmp_path, provider="vllm", model="qwen")
        second = store.create_session(workspace=tmp_path, provider="vllm", model="qwen")
        store.close()

        main(["sessions"])
        out = capsys.readouterr().out
        assert out.index(second.id) < out.index(first.id)


class TestMemories:
    @pytest.fixture
    def db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / "share" / "memories.db"
        monkeypatch.setattr(loader, "memory_db_path", lambda: path)
        return path

    def _store(self, path: Path) -> MemoryStore:
        store = MemoryStore(path)
        store.add(content="we use uv, not pip", type=MemoryType.DECISION, scope="global")
        return store

    def test_listing_on_a_fresh_machine_creates_no_database(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["memories"]) == 0
        assert "nothing stored yet" in capsys.readouterr().out
        assert not db.exists(), "listing memories must not create a database"

    def test_list_shows_the_columns_needed_to_act_on_a_row(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = self._store(db)
        store.close()
        assert main(["memories", "list"]) == 0
        out = capsys.readouterr().out
        for column in ("id", "type", "created", "scope", "content"):
            assert column in out
        assert "we use uv, not pip" in out
        assert "decision" in out
        assert str(db) in out

    def test_a_long_scope_does_not_swallow_the_content_column(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real scope is a normalised remote, which is longer than the column.

        ``str:<N`` pads but never truncates, so a scope over the column width
        runs straight into the content and the table stops being a table. Every
        other test here files under ``global``, six characters wide, which is why
        none of them noticed.
        """
        store = MemoryStore(db)
        scope = "repo:github.com/thadreber-web/bevel"
        store.add(content="we use uv, not pip", type=MemoryType.DECISION, scope=scope)
        store.close()
        assert len(scope) > 34, "the fixture must be wider than the scope column"

        assert main(["memories", "list"]) == 0
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if "we use uv" in line)
        assert f"{scope}we use" not in row, "the scope column ran into the content column"
        assert "repo:github.com/thadreber-web/..." in row, "the scope was dropped, not elided"
        assert "we use uv, not pip" in row
        # The columns are fixed width, so the content always starts at the same
        # offset whether the scope was padded or elided. Asserting the offset is
        # what catches an elision one character too wide: ``f"{scope[:32]}..."``
        # is 35 characters in a 34-wide column and merely *touches* the content.
        header = next(line for line in out.splitlines() if "content" in line)
        assert row.index("we use uv, not pip") == header.index("content")

    def test_a_superseded_memory_is_hidden_from_list_but_still_readable(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = self._store(db)
        old = store.list_memories()[0].id
        store.add(
            content="we use uv, and pip is fine too", type=MemoryType.DECISION, scope="global"
        )
        new = store.list_memories()[0].id
        store.supersede(old, by=new)
        store.close()

        assert main(["memories", "list"]) == 0
        out = capsys.readouterr().out
        assert "we use uv, not pip" not in out
        assert "superseded" in out

        assert main(["memories", "show", str(old)]) == 0
        shown = capsys.readouterr().out
        assert "we use uv, not pip" in shown
        assert f"superseded by #{new}" in shown

    def test_show_reports_the_vector_and_its_model(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = MemoryStore(db)
        store.add(
            content="the release is tagged from main",
            type=MemoryType.FACT,
            scope="global",
            vector=(1.0, 0.0),
            vector_model="bge-m3",
        )
        memory_id = store.list_memories()[0].id
        store.close()

        assert main(["memories", "show", str(memory_id)]) == 0
        out = capsys.readouterr().out
        assert "bge-m3" in out
        assert "the release is tagged from main" in out

    def test_supersede_links_the_two_rows(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = self._store(db)
        old = store.list_memories()[0].id
        store.add(content="uv and pip both work", type=MemoryType.DECISION, scope="global")
        new = store.list_memories()[0].id
        store.close()

        assert main(["memories", "supersede", str(old), str(new)]) == 0
        assert f"#{old} is superseded by #{new}" in capsys.readouterr().out

        store = MemoryStore(db)
        try:
            assert store.get(old) is not None, "supersede deleted the row"
            assert store.get(old).superseded_by == new  # type: ignore[union-attr]
        finally:
            store.close()

    def test_superseding_twice_fails_loudly(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = self._store(db)
        old = store.list_memories()[0].id
        replacement = store.add(content="a later belief", type=MemoryType.FACT, scope="global").id
        store.close()

        assert main(["memories", "supersede", str(old), str(replacement)]) == 0
        capsys.readouterr()
        assert main(["memories", "supersede", str(old), str(replacement)]) == 2
        assert "already superseded" in capsys.readouterr().err

    def test_an_unknown_id_is_a_failure_not_an_empty_success(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._store(db).close()
        assert main(["memories", "show", "9999"]) == 2
        assert "no memory with id 9999" in capsys.readouterr().err

    def test_a_non_numeric_id_is_rejected(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._store(db).close()
        assert main(["memories", "show", "abc"]) == 2
        assert "not a memory id" in capsys.readouterr().err

    def test_supersede_without_two_ids_is_a_usage_error(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._store(db).close()
        assert main(["memories", "supersede", "1"]) == 2
        assert "<old-id> <new-id>" in capsys.readouterr().err

    def test_an_unimplemented_action_is_rejected_by_the_parser(
        self, db: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No delete exists, and `bevel memories delete` must not look like it does."""
        with pytest.raises(SystemExit) as excinfo:
            main(["memories", "delete", "1"])
        assert excinfo.value.code == 2
        assert "invalid choice" in capsys.readouterr().err
        assert not db.exists(), "a rejected action must not create a database"


class TestResume:
    @pytest.fixture
    def launched(self, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
        opened: list[str | None] = []

        def fake_run_tui(workspace: Path | str | None = None, resume: str | None = None) -> int:
            del workspace
            opened.append(resume)
            return 0

        monkeypatch.setattr("bevel.tui.app.run_tui", fake_run_tui)
        return opened

    def test_a_named_session_opens_the_tui_on_that_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        launched: list[str | None],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        info = store.create_session(workspace=tmp_path, provider="vllm", model="qwen")
        store.close()
        assert main(["resume", info.id]) == 0
        assert launched == [info.id]

    def test_a_prefix_is_enough(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        launched: list[str | None],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        info = store.create_session(workspace=tmp_path, provider="vllm", model="qwen")
        store.close()
        assert main(["resume", info.id[:4]]) == 0
        assert launched == [info.id]

    def test_an_unknown_id_fails_naming_what_was_asked_for(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        launched: list[str | None],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        store.create_session(workspace=tmp_path, provider="v", model="m")
        store.close()
        assert main(["resume", "nosuchid"]) == 2
        err = capsys.readouterr().err
        assert "no session id starts with 'nosuchid'" in err
        assert launched == []

    def test_an_ambiguous_prefix_refuses_rather_than_guessing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        launched: list[str | None],
    ) -> None:
        """Resuming the wrong session silently rewrites an unrelated history.

        The ids are forced to share a prefix rather than left to chance: with 64
        random 16-hex-char ids there is a ~1.8% chance that none of them share
        any prefix, which made this test fail with ``StopIteration`` for reasons
        that had nothing to do with the behaviour under test.
        """
        db = _pin_db(tmp_path, monkeypatch)
        counter = iter(range(64))
        monkeypatch.setattr(
            session_store, "token_hex", lambda _n: f"a1b2{next(counter):012x}", raising=True
        )
        store = SessionStore(db)
        ids = [
            store.create_session(workspace=tmp_path, provider="v", model="m").id for _ in range(64)
        ]
        store.close()
        assert len(set(ids)) == 64
        shared = next(
            prefix
            for prefix in (ids[0][:i] for i in range(1, len(ids[0]) + 1))
            if sum(1 for session_id in ids if session_id.startswith(prefix)) > 1
        )
        assert main(["resume", shared]) == 2
        err = capsys.readouterr().err
        assert "ambiguous" in err
        assert launched == []

    def test_no_id_resumes_the_most_recent_session_for_this_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        launched: list[str | None],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        store.create_session(workspace=tmp_path, provider="v", model="m")
        newest = store.create_session(workspace=tmp_path, provider="v", model="m")
        store.close()
        monkeypatch.chdir(tmp_path)
        assert main(["resume"]) == 0
        assert launched == [newest.id]

    def test_no_id_and_no_stored_session_fails_with_a_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        launched: list[str | None],
    ) -> None:
        _pin_db(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        assert main(["resume"]) == 2
        err = capsys.readouterr().err
        assert "no sessions have been stored yet" in err
        assert launched == []

    def test_a_session_from_another_directory_is_not_the_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        launched: list[str | None],
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        store.create_session(workspace=tmp_path / "elsewhere", provider="v", model="m")
        store.close()
        here = tmp_path / "here"
        here.mkdir()
        monkeypatch.chdir(here)
        assert main(["resume"]) == 2
        err = capsys.readouterr().err
        assert "no stored session for" in err
        assert "bevel sessions" in err
        assert launched == []


class TestTodos:
    """``bevel todos [id]`` -- the only user-facing surface for the list this milestone adds.

    A tracker nobody can read is a fake, so these assert on what the terminal
    actually shows: the tasks, in order, with their statuses, and the count.
    """

    def _seed(
        self,
        db: Path,
        workspace: Path,
        *items: tuple[str, str],
    ) -> str:
        sessions = SessionStore(db)
        info = sessions.create_session(workspace=workspace, provider="v", model="m")
        sessions.close()
        store = TodoStore(db)
        store.session_id = info.id
        store.replace(
            [Todo(content=content, status=TodoStatus(status)) for content, status in items]
        )
        store.close()
        return info.id

    def test_reading_on_a_fresh_machine_creates_no_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        assert main(["todos"]) == 0
        assert "no sessions have been stored yet" in capsys.readouterr().out
        assert not db.exists(), "reading todos must not create a database"

    def test_the_list_is_printed_in_order_with_its_statuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        self._seed(
            db,
            tmp_path,
            ("read the parser", "done"),
            ("fix the grammar", "in_progress"),
            ("add a test", "pending"),
        )
        monkeypatch.chdir(tmp_path)
        assert main(["todos"]) == 0
        out = capsys.readouterr().out
        assert "[x] read the parser" in out
        assert "[>] fix the grammar" in out
        assert "[ ] add a test" in out
        assert (
            out.index("read the parser") < out.index("fix the grammar") < out.index("add a test")
        ), "the list came out reordered; the order is the plan"
        assert "1 of 3 done" in out

    def test_a_named_session_or_prefix_reads_that_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        mine = self._seed(db, tmp_path, ("mine", "pending"))
        self._seed(db, tmp_path, ("theirs", "done"))
        capsys.readouterr()

        assert main(["todos", mine[:6]]) == 0
        out = capsys.readouterr().out
        assert "mine" in out
        assert "theirs" not in out, "reading one session showed another's plan"
        assert mine in out

    def test_the_default_is_the_most_recent_session_for_this_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        here = tmp_path / "here"
        here.mkdir()
        self._seed(db, tmp_path / "elsewhere", ("not mine", "pending"))
        mine = self._seed(db, here, ("mine", "pending"))
        capsys.readouterr()
        monkeypatch.chdir(here)

        assert main(["todos"]) == 0
        out = capsys.readouterr().out
        assert mine in out
        assert "not mine" not in out

    def test_a_session_with_no_todos_says_so_rather_than_printing_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        sessions = SessionStore(db)
        info = sessions.create_session(workspace=tmp_path, provider="v", model="m")
        sessions.close()
        assert main(["todos", info.id]) == 0
        out = capsys.readouterr().out
        assert "no todos stored" in out
        assert info.id in out

    def test_an_unknown_id_fails_naming_what_was_asked_for(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        self._seed(db, tmp_path, ("anything", "pending"))
        assert main(["todos", "nosuchid"]) == 2
        assert "no session id starts with 'nosuchid'" in capsys.readouterr().err

    def test_a_session_from_another_directory_is_not_the_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        db = _pin_db(tmp_path, monkeypatch)
        store = SessionStore(db)
        store.create_session(workspace=tmp_path / "elsewhere", provider="v", model="m")
        store.close()
        here = tmp_path / "here"
        here.mkdir()
        monkeypatch.chdir(here)
        assert main(["todos"]) == 2
        err = capsys.readouterr().err
        assert "no stored session for" in err
        assert "bevel sessions" in err

    def test_a_cleared_list_reads_back_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The model clears its list by sending an empty one; the CLI must agree."""
        db = _pin_db(tmp_path, monkeypatch)
        session_id = self._seed(db, tmp_path, ("was here", "done"))
        store = TodoStore(db)
        store.session_id = session_id
        store.replace(())
        store.close()
        capsys.readouterr()

        assert main(["todos", session_id]) == 0
        assert "no todos stored" in capsys.readouterr().out

    def test_the_subcommand_takes_positionals_only(self) -> None:
        """A flag here would have to be read by this milestone, and none is."""
        action = next(
            action
            for action in build_parser()._subparsers._group_actions  # type: ignore[union-attr]
            if action.choices and "todos" in action.choices
        )
        parser = action.choices["todos"]
        options = {option for item in parser._actions for option in item.option_strings}
        assert options == {"-h", "--help"}


class TestUndo:
    def test_undo_without_a_repository_refuses_with_a_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert main(["undo"]) == 1
        err = capsys.readouterr().err
        assert err.startswith("bevel undo:")
        assert "nothing restored" in err

    def test_undo_restores_the_last_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Over a real repository, because restore's semantics *are* git's."""

        def git(*args: str) -> None:
            subprocess.run(
                ["git", "-c", "user.name=bevel", "-c", "user.email=b@l", *args],
                cwd=tmp_path,
                check=True,
                capture_output=True,
            )

        git("init", "-q")
        (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
        git("add", "a.py")
        git("commit", "-qm", "base")
        asyncio.run(snapshot(tmp_path, session="s1"))
        (tmp_path / "a.py").write_text("one\ntwo\n", encoding="utf-8")
        (tmp_path / "extra.py").write_text("new\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        assert main(["undo"]) == 0
        out = capsys.readouterr().out
        assert (tmp_path / "a.py").read_text(encoding="utf-8") == "one\n"
        assert "restored" in out
        # A file created after the checkpoint is left alone unless asked for.
        assert (tmp_path / "extra.py").exists()
        assert "left 1 untracked file(s) in place" in out


class TestDefaultInvocation:
    def test_bare_invocation_launches_the_tui(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Bare `bevel` opens the TUI -- and does not start a real one here."""
        launched: list[bool] = []

        def fake_run_tui() -> int:
            launched.append(True)
            return 0

        monkeypatch.setattr("bevel.tui.app.run_tui", fake_run_tui)
        assert main([]) == 0
        assert launched == [True]
        assert capsys.readouterr().err == ""

    def test_the_tui_is_imported_lazily(self) -> None:
        """`doctor` and `--version` must not pay textual's ~1.3s import."""
        probe = subprocess.run(
            [sys.executable, "-c", "import bevel.cli, sys; print('textual' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == "False"


class TestMcp:
    """``bevel mcp list`` -- the only way to see what the config actually does.

    These spawn ``tests/fixtures/mcp_server.py`` for real. A list command tested
    against a stubbed transport would print whatever the stub was told to, which
    is exactly the kind of confident lie this project refuses to ship: the
    command's whole job is to tell you whether a server is really up.
    """

    @pytest.fixture(autouse=True)
    def _reap_fixtures(self) -> Iterator[None]:
        """Kill any HTTP fixture a test started, whatever the test did.

        ``bevel mcp list`` closes its own transports, but a fixture is a process
        this file started, and a leaked one on Windows outlives the test run.
        """
        self._procs: list[subprocess.Popen[bytes]] = []
        yield
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def _configure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        servers: str,
    ) -> Path:
        """Write a project config with the given ``[mcp.servers.*]`` tables."""
        for name in ("BEVEL_BASE_URL", "BEVEL_MODEL", "BEVEL_PROFILE", "BEVEL_CONFIG"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(loader, "config_home", lambda: tmp_path / "empty-config-home")
        project = tmp_path / "project"
        project.mkdir(parents=True, exist_ok=True)
        (project / "bevel.toml").write_text(
            "[profile.default]\n"
            'base_url = "https://models.test/v1"\n'
            'model = "test-model"\n'
            f"\n{servers}\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(project)
        return project

    def _fixture_table(self, name: str = "fixture", **fields: object) -> str:
        # ``json.dumps`` rather than an f-string: a Windows path in a TOML basic
        # string needs its backslashes escaped, and JSON already escapes exactly
        # the way TOML does.
        lines = [
            f"[mcp.servers.{name}]",
            f"command = {json.dumps(sys.executable)}",
            f"args = [{json.dumps(str(FIXTURE))}]",
        ]
        lines.extend(f"{key} = {value}" for key, value in fields.items())
        return "\n".join(lines)

    def test_mcp_is_no_longer_a_pending_command(self) -> None:
        """The honesty rule: a command that works must not announce itself as unbuilt."""
        assert "mcp" not in _PENDING

    def test_the_subcommand_takes_only_positionals(self) -> None:
        """A ``--flag`` here would have to be read by this milestone."""
        subparser = next(
            action.choices["mcp"]
            for action in build_parser()._actions
            if action.choices and "mcp" in action.choices
        )
        options = {option for action in subparser._actions for option in action.option_strings}
        assert options == {"-h", "--help"}

    def test_no_action_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["mcp"]) == 2
        assert "list" in capsys.readouterr().err

    def test_an_unknown_action_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["mcp", "bogus"])
        assert excinfo.value.code == 2

    def test_a_server_that_handshakes_is_reported_up_with_its_tools(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, self._fixture_table())
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "fixture" in out
        assert "enabled" in out
        assert "up" in out
        assert "echo" in out
        assert "read_note" in out
        assert str(FIXTURE) in out, "the command is what a user checks when a server is down"

    def test_a_disabled_server_is_reported_disabled_and_never_spawned(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Silently skipping it would leave the user wondering why it never appears."""
        self._configure(tmp_path, monkeypatch, self._fixture_table(enabled="false"))
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "disabled" in out
        assert "up" not in out.split(), "a disabled server must not be reported as running"

    def test_a_server_that_cannot_start_is_reported_plainly_and_still_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Reporting a down server faithfully *is* success; the command's job is to tell the truth."""
        self._configure(
            tmp_path,
            monkeypatch,
            self._fixture_table()
            + '\n[mcp.servers.broken]\ncommand = "definitely-not-a-program"\n',
        )
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "broken" in out
        assert "down" in out
        assert "definitely-not-a-program" in out
        assert "fixture" in out, "one dead server must not hide the healthy one"

    def test_the_environment_values_are_never_printed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Same leak class as ``api_key_env``: a token in a config is not print material.

        ``bevel mcp list`` output ends up in a terminal someone screenshots and
        in a bug report someone pastes, so the values never leave the process.
        """
        self._configure(
            tmp_path,
            monkeypatch,
            self._fixture_table(env='{TOKEN = "hunter2-super-secret"}'),
        )
        assert main(["mcp", "list"]) == 0
        captured = capsys.readouterr()
        out = captured.out + captured.err
        assert "hunter2-super-secret" not in out
        assert "TOKEN" not in out, "naming the variable is fine, printing its value is not"

    def test_an_http_server_is_reported_the_same_way_a_stdio_one_is(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """One list, two transports: the row must not change shape.

        The fixture is spawned in HTTP mode by the test, but the server is
        started by ``bevel mcp list`` from a config holding only ``url``, so this
        is the real command path rather than a rendering of a hand-made report.
        """
        url = self._serve_http(tmp_path)
        self._configure(tmp_path, monkeypatch, self._http_table(url))
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "remote" in out
        assert "enabled" in out
        assert "up" in out
        assert "echo" in out, "an HTTP server's tools must be listed like a stdio server's"
        assert url in out, "the URL is what a user checks when the server is down"

    def test_an_http_server_that_never_handshakes_is_reported_down_and_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The not-handshaked case, which is the one that actually happens.

        A port bound by a socket nobody accepts on is what "the URL is wrong"
        looks like once the name has resolved, and it is the closest thing to a
        dead server that does not depend on DNS or on someone else's machine.
        """
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(0)
        port = listener.getsockname()[1]
        listener.close()
        self._configure(
            tmp_path,
            monkeypatch,
            self._http_table(f"http://127.0.0.1:{port}") + "\n" + self._fixture_table(),
        )
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "remote" in out
        assert "down" in out
        assert f"http://127.0.0.1:{port}" in out
        assert "fixture" in out, "one dead server must not hide the healthy one"
        assert "1 of 2 server(s) answered" in out

    def _serve_http(self, tmp_path: Path) -> str:
        """Spawn the fixture in HTTP mode for the duration of the test; return its URL."""
        port_file = tmp_path / "port"
        proc = subprocess.Popen(
            [sys.executable, str(FIXTURE), "--http", "--port-file", str(port_file)],
            cwd=str(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._procs.append(proc)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if port_file.exists() and port_file.read_text(encoding="utf-8").strip():
                return f"http://127.0.0.1:{port_file.read_text(encoding='utf-8').strip()}"
            if proc.poll() is not None:
                raise AssertionError(f"the fixture exited early with {proc.returncode}")
            time.sleep(0.02)
        raise AssertionError("the fixture never wrote its port file")

    def _http_table(self, url: str, **fields: object) -> str:
        lines = [
            "[mcp.servers.remote]",
            f"url = {json.dumps(url)}",
            "timeout = 15.0",
        ]
        lines.extend(f"{key} = {value}" for key, value in fields.items())
        return "\n".join(lines)

    def test_no_servers_configured_says_so_and_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, "")
        assert main(["mcp", "list"]) == 0
        out = capsys.readouterr().out
        assert "no MCP servers configured" in out
        assert "bevel.toml" in out, "the message has to say where a server would be configured"

    def test_an_unusable_config_exits_non_zero_naming_the_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A typo'd key is a real failure, unlike a down server: nothing was reported faithfully."""
        self._configure(tmp_path, monkeypatch, self._fixture_table(nonsense="1"))
        assert main(["mcp", "list"]) == 2
        err = capsys.readouterr().err
        assert "nonsense" in err
        assert "bevel.toml" in err

    # --- bevel mcp resources <server> ---------------------------------------

    def test_resources_lists_the_uris_a_live_server_offers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The human path and the model path share one code path, so both must really ask."""
        self._configure(tmp_path, monkeypatch, self._fixture_table())
        assert main(["mcp", "resources", "fixture"]) == 0
        out = capsys.readouterr().out
        assert "bevel://fixture/notes" in out
        assert "notes" in out
        assert "The fixture's one resource." in out
        assert "text/plain" in out

    def test_resources_on_a_server_with_no_resources_says_so_and_exits_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An honest empty list is an answer, not a failure -- and not a blank line."""
        self._configure(
            tmp_path,
            monkeypatch,
            self._fixture_table_with_flags("--resources-list", "empty"),
        )
        assert main(["mcp", "resources", "fixture"]) == 0
        out = capsys.readouterr().out
        assert "no resources" in out.lower()
        assert out.strip(), "a blank line cannot be told apart from a broken command"

    def test_resources_on_a_down_server_fails_naming_the_reason(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Down is not empty. Printing an empty list here would be the worst answer."""
        self._configure(
            tmp_path,
            monkeypatch,
            '[mcp.servers.broken]\ncommand = "definitely-not-a-program"\n',
        )
        assert main(["mcp", "resources", "broken"]) == 2
        err = capsys.readouterr().err
        assert "definitely-not-a-program" in err
        assert "no resources" not in err.lower()

    def test_resources_on_a_disabled_server_says_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, self._fixture_table(enabled="false"))
        assert main(["mcp", "resources", "fixture"]) == 2
        err = capsys.readouterr().err
        assert "disabled" in err
        assert "no resources" not in err.lower()

    def test_resources_on_an_unknown_server_names_the_ones_that_are(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, self._fixture_table())
        assert main(["mcp", "resources", "typo"]) == 2
        err = capsys.readouterr().err
        assert "typo" in err
        assert "fixture" in err, "the user needs to know what they could have typed"

    def test_resources_without_a_server_name_is_a_usage_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, self._fixture_table())
        assert main(["mcp", "resources"]) == 2
        assert "which server" in capsys.readouterr().err

    def test_resources_with_no_servers_configured_says_where_to_add_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._configure(tmp_path, monkeypatch, "")
        assert main(["mcp", "resources", "fixture"]) == 2
        err = capsys.readouterr().err
        assert "no MCP servers configured" in err
        assert "bevel.toml" in err

    def _fixture_table_with_flags(self, *flags: str) -> str:
        """The fixture's table with extra argv, for driving one of its failure modes."""
        lines = [
            "[mcp.servers.fixture]",
            f"command = {json.dumps(sys.executable)}",
            f"args = [{', '.join(json.dumps(str(item)) for item in (FIXTURE, *flags))}]",
        ]
        return "\n".join(lines)


class TestUnknownInput:
    def test_unknown_command_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["frobnicate"])
        assert excinfo.value.code == 2

    def test_unknown_flag_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--model", "llama-70b"])
        assert excinfo.value.code == 2

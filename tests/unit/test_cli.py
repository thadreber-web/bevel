"""Tests for the CLI contract.

The rules these enforce are the project's honesty rules: nothing may be
accepted and then silently ignored, and unimplemented commands must fail
loudly while naming the milestone that delivers them.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from bevel import __version__
from bevel.cli import _PENDING, build_parser, main
from bevel.config import loader
from bevel.core.checkpoint import snapshot
from bevel.core.messages import Message
from bevel.core.session import store as session_store
from bevel.core.session.store import SessionStore


def _pin_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the session store at a file in tmp_path, as the TUI tests do."""
    path = tmp_path / "share" / "sessions.db"
    monkeypatch.setattr(loader, "session_db_path", lambda: path)
    return path


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


class TestUnknownInput:
    def test_unknown_command_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["frobnicate"])
        assert excinfo.value.code == 2

    def test_unknown_flag_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--model", "llama-70b"])
        assert excinfo.value.code == 2

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
from bevel.core.checkpoint import snapshot


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

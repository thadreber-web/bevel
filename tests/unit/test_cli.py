"""Tests for the CLI contract.

The rules these enforce are the project's honesty rules: nothing may be
accepted and then silently ignored, and unimplemented commands must fail
loudly while naming the milestone that delivers them.
"""

from __future__ import annotations

import pytest

from bevel import __version__
from bevel.cli import _PENDING, build_parser, main


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


class TestDefaultInvocation:
    def test_bare_invocation_explains_the_tui_is_missing(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 2
        err = capsys.readouterr().err
        assert "m07" in err
        assert "bevel doctor" in err


class TestUnknownInput:
    def test_unknown_command_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["frobnicate"])
        assert excinfo.value.code == 2

    def test_unknown_flag_is_rejected(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main(["--model", "llama-70b"])
        assert excinfo.value.code == 2

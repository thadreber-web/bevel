"""Command-line entry point.

``bevel`` with no subcommand launches the TUI. Flags and subcommands are added
by the milestone that implements them: an accepted-but-unread flag would let
``bevel --model x`` quietly do nothing, which is the quiet lie this project
exists to avoid. Subcommands that are planned but not built are listed in
``_PENDING`` and fail loudly, naming the milestone that delivers them.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from . import __version__
from ._capability import CapabilityReport, probe
from .core.checkpoint import restore
from .core.session.store import SessionStore

#: Subcommands that are planned but not built yet, mapped to the milestone
#: that delivers them. Kept in one place so ``--help`` and the stub message
#: cannot drift apart.
_PENDING: dict[str, str] = {
    "memory": "m09 (long-term memory)",
    "mcp": "m10 (MCP client)",
    "pr": "m11 (git and PR operations)",
    "sandbox": "m12 (sandbox backends)",
    "eval": "m13 (eval harness)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bevel",
        description="A terminal coding agent for your own models (vLLM, llama.cpp).",
        epilog="Run 'bevel' with no subcommand to open the TUI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bevel {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser("doctor", help="report environment, endpoints and sandbox capability")
    subparsers.add_parser("undo", help="rewind the working tree to the last checkpoint")
    subparsers.add_parser("sessions", help="list recent sessions")
    resume = subparsers.add_parser("resume", help="reopen a session and continue it")
    resume.add_argument(
        "session",
        nargs="?",
        help="session id or prefix (default: the most recent session for this directory)",
    )
    for name in _PENDING:
        subparsers.add_parser(name, help=f"(planned) {name}")
    return parser


def _render_doctor(report: CapabilityReport) -> str:
    lines = [
        "bevel doctor",
        "=" * 12,
        f"  platform        {report.platform} {report.release} ({report.machine})",
        f"  python          {report.python_version}",
        f"  in container    {'yes' if report.in_container else 'no'}",
        f"  best tier       {report.best_tier} "
        "(0=policy-only, 1=kernel, 2=os-container; verified at runtime by self-test)",
        "",
        "  sandbox capabilities",
    ]
    for cap in report.capabilities:
        mark = "yes" if cap.available else "no "
        suffix = f"  [{cap.error}]" if cap.error else ""
        lines.append(f"    [{mark}] {cap.name:<17} {cap.detail}{suffix}")
    lines += [
        "",
        "  Tier 2 (Windows Sandbox / bwrap) is opt-in and is not used for ordinary",
        "  tool calls: a 10-30s cold start per command is unusable, and the",
        "  container is disposable while an agent must write real edits. It is",
        "  enabled by config when m12 delivers the container backends.",
        "",
        "  Static probing only. bevel reports the enforcement level it *verified*,",
        "  which can be lower than 'best tier' above.",
    ]
    return "\n".join(lines)


def _undo(out: IO[str], err: IO[str]) -> int:
    """Rewind the working tree to the most recent checkpoint.

    The restore API refuses rather than guessing -- no repository, no
    checkpoints, or a state it cannot safely reconcile all come back as a
    refusal with a reason. The refusal goes to stderr because it is a failure;
    the rest of the report is what happened, so it goes to stdout.
    """
    result = asyncio.run(restore(Path.cwd()))
    if not result.ok or result.commit is None:
        err.write(f"bevel undo: {result.error or 'nothing was restored'}\n")
        return 1
    out.write(f"bevel undo: restored {result.commit[:8]}.\n")
    if result.removed:
        out.write(f"  removed {len(result.removed)} file(s) created after the checkpoint\n")
    if result.untracked:
        out.write(f"  left {len(result.untracked)} untracked file(s) in place\n")
    return 0


def _not_implemented(command: str, stream: IO[str]) -> int:
    stream.write(f"bevel {command}: not implemented yet -- planned for {_PENDING[command]}.\n")
    return 2


def _stamp(value: str) -> str:
    """``2026-02-06T12:34:56+00:00`` -> ``2026-02-06 12:34``, which is what fits a table."""
    date, _, rest = value.partition("T")
    return f"{date} {rest[:5]}" if rest else value


def _sessions(out: IO[str]) -> int:
    """List recent sessions, newest first.

    The row is what a user needs in order to *choose*: enough of an id to type
    back, the workspace it belonged to, when it ran, and what it cost.
    """
    from .config.loader import session_db_path  # noqa: PLC0415

    store = SessionStore(session_db_path())
    try:
        # Reading a file that was never written would create it; `bevel sessions`
        # on a fresh machine must leave nothing behind.
        sessions = () if not store.db_path.exists() else store.list_sessions()
        if not sessions:
            out.write(f"bevel sessions: no sessions stored yet ({store.db_path}).\n")
            return 0
        out.write(
            f"{'id':<17}{'started':<17}{'last activity':<16}"
            f"{'turns':>6}{'prompt':>9}{'completion':>12}  workspace\n"
        )
        for info in sessions:
            out.write(
                f"{info.id:<17}{_stamp(info.started_at):<17}"
                f"{_stamp(info.last_activity_at):<16}"
                f"{info.turns:>6}{info.prompt_tokens:>9,}{info.completion_tokens:>12,}"
                f"  {info.workspace}\n"
            )
        out.write(f"\n  {len(sessions)} session(s) in {store.db_path}\n")
        return 0
    finally:
        store.close()


def _resume(named: str | None) -> int:
    """Reopen a session in the TUI: a named one, or the most recent for here."""
    from .config.loader import session_db_path  # noqa: PLC0415

    store = SessionStore(session_db_path())
    try:
        if not store.db_path.exists():
            sys.stderr.write(f"bevel resume: no sessions have been stored yet ({store.db_path}).\n")
            return 2
        if named is None:
            session_id = store.latest_session_id(Path.cwd())
            if session_id is None:
                sys.stderr.write(
                    f"bevel resume: no stored session for {Path.cwd()}."
                    " Name one with 'bevel resume <id>' (see 'bevel sessions').\n"
                )
                return 2
        else:
            try:
                session_id = store.resolve(named).id
            except ValueError as exc:
                sys.stderr.write(f"bevel resume: {exc}\n")
                return 2
    finally:
        store.close()

    # Imported here rather than at module scope, as with the TUI itself.
    from .tui.app import run_tui  # noqa: PLC0415

    return run_tui(resume=session_id)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never calls ``sys.exit``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        sys.stdout.write(_render_doctor(probe()) + "\n")
        return 0

    if args.command == "undo":
        return _undo(sys.stdout, sys.stderr)

    if args.command == "sessions":
        return _sessions(sys.stdout)

    if args.command == "resume":
        return _resume(args.session)

    if args.command in _PENDING:
        return _not_implemented(args.command, sys.stderr)

    if args.command is not None:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command: {args.command}")

    # Imported here rather than at module scope: textual costs ~1.3s to import,
    # and `bevel doctor` and `bevel --version` must not pay for it.
    from .tui.app import run_tui  # noqa: PLC0415

    return run_tui()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

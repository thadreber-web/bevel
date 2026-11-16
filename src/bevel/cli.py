"""Command-line entry point.

``bevel`` with no subcommand launches the TUI. Flags and subcommands are added
by the milestone that implements them: an accepted-but-unread flag would let
``bevel --model x`` quietly do nothing, which is the quiet lie this project
exists to avoid. Subcommands that are planned but not built are listed in
``_PENDING`` and fail loudly, naming the milestone that delivers them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import IO

from . import __version__
from ._capability import CapabilityReport, probe

#: Subcommands that are planned but not built yet, mapped to the milestone
#: that delivers them. Kept in one place so ``--help`` and the stub message
#: cannot drift apart.
_PENDING: dict[str, str] = {
    "sessions": "m08 (session persistence and resume)",
    "resume": "m08 (session persistence and resume)",
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


def _not_implemented(command: str, stream: IO[str]) -> int:
    stream.write(f"bevel {command}: not implemented yet -- planned for {_PENDING[command]}.\n")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never calls ``sys.exit``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        sys.stdout.write(_render_doctor(probe()) + "\n")
        return 0

    if args.command in _PENDING:
        return _not_implemented(args.command, sys.stderr)

    if args.command is not None:  # pragma: no cover - argparse rejects unknown commands
        parser.error(f"unknown command: {args.command}")

    sys.stderr.write(
        "The bevel TUI is not implemented yet -- planned for m07 (Textual TUI).\n"
        "Try 'bevel doctor' to see what this machine supports.\n"
    )
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

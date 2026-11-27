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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

from . import __version__
from ._capability import CapabilityReport, probe
from .core.checkpoint import restore
from .core.memory.store import MemoryStore
from .core.session.store import SessionInfo, SessionStore, TodoStatus, TodoStore

if TYPE_CHECKING:
    from .config.schema import McpServerConfig
    from .mcp.servers import ServerReport

#: Subcommands that are planned but not built yet, mapped to the milestone
#: that delivers them. Kept in one place so ``--help`` and the stub message
#: cannot drift apart.
_PENDING: dict[str, str] = {
    "pr": "m11b (PR create/update via gh)",
    "sandbox": "m12 (sandbox backends)",
    "eval": "m13 (eval harness)",
}

#: What ``bevel mcp`` accepts. ``resources`` is m10f and shares
#: :func:`~bevel.mcp.resources.list_resources` with the model-facing tool, so a
#: human and a model ask the server the same question the same way. There is no
#: ``read`` here: a resource is worth looking at once the model has listed them,
#: and a CLI that printed the whole body would be the one place bevel ignores its
#: own cap. There is no ``start`` or ``restart`` because a server's lifetime
#: belongs to a run, not to a CLI command that would leave a process behind when
#: the shell exited.
_MCP_ACTIONS: tuple[str, ...] = ("list", "resources")

#: What ``bevel memories`` accepts. All three are built, so anything else is a
#: usage error argparse rejects with exit 2 rather than a stub that pretends.
#: There is no ``delete`` and none is planned: a wrong memory is superseded, so
#: the chain of what bevel believed stays readable.
_MEMORY_ACTIONS: tuple[str, ...] = ("list", "show", "supersede")

#: How each status is drawn by ``bevel todos``.
_TODO_MARKS: dict[TodoStatus, str] = {
    TodoStatus.PENDING: " ",
    TodoStatus.IN_PROGRESS: ">",
    TodoStatus.DONE: "x",
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
    subparsers.add_parser("doctor", help="report platform, Python and sandbox capability")
    subparsers.add_parser("undo", help="rewind the working tree to the last checkpoint")
    subparsers.add_parser("sessions", help="list recent sessions")
    memories = subparsers.add_parser("memories", help="inspect and correct long-term memory")
    # Positionals only. A `--flag` here would have to be read by this milestone,
    # and TestParser rejects an option the current code does not act on.
    memories.add_argument("action", nargs="?", choices=_MEMORY_ACTIONS, help="what to do")
    memories.add_argument("args", nargs="*", metavar="ARG", help="memory id(s) the action needs")
    resume = subparsers.add_parser("resume", help="reopen a session and continue it")
    resume.add_argument(
        "session",
        nargs="?",
        help="session id or prefix (default: the most recent session for this directory)",
    )
    todos = subparsers.add_parser("todos", help="show a session's task list")
    # Positionals only, as with `memories`: a `--flag` here would have to be read
    # by this milestone, and TestParser rejects an option the code does not act on.
    todos.add_argument(
        "session",
        nargs="?",
        help="session id or prefix (default: the most recent session for this directory)",
    )
    mcp = subparsers.add_parser("mcp", help="inspect the configured MCP servers")
    mcp.add_argument("action", nargs="?", choices=_MCP_ACTIONS, help="what to do")
    # Positional, not an option: TestMcp rejects an option this code would not
    # act on, and `resources` is the only action that needs a target.
    mcp.add_argument(
        "server",
        nargs="?",
        help="server name from the [mcp.servers.<name>] table (required for 'resources')",
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
        "(0=policy-only, 1=kernel, 2=os-container; what your OS could enforce)",
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
        "  Static probing only. No sandbox backend ships yet: bevel enforces the T0",
        "  policy jail, whatever 'best tier' above says. m12 adds the backends and",
        "  the self-test that would justify a higher number.",
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


def _memory_id(raw: str, err: IO[str]) -> int | None:
    try:
        return int(raw)
    except ValueError:
        err.write(
            f"bevel memories: {raw!r} is not a memory id. Ids are the numbers in the first column.\n"
        )
        return None


def _memories(  # noqa: PLR0911 - one exit per argument shape; a dispatcher reads better than nesting
    action: str | None, args: Sequence[str], out: IO[str], err: IO[str]
) -> int:
    """List, read, and correct what bevel has stored.

    Reads never create the database, so the command on a fresh machine leaves
    nothing behind. There is no delete: a wrong memory is superseded, which
    keeps the chain auditable, and ``show`` can still read what was replaced.
    """
    from .config.loader import memory_db_path  # noqa: PLC0415

    store = MemoryStore(memory_db_path())
    try:
        if not store.db_path.exists():
            out.write(f"bevel memories: nothing stored yet ({store.db_path}).\n")
            return 0
        if action in (None, "list"):
            return _memories_list(store, out)
        if action == "show":
            if len(args) != 1:
                err.write("bevel memories show: expected exactly one memory id.\n")
                return 2
            memory_id = _memory_id(args[0], err)
            return 2 if memory_id is None else _memories_show(store, memory_id, out, err)
        if len(args) != 2:
            err.write("bevel memories supersede: expected <old-id> <new-id>.\n")
            return 2
        old = _memory_id(args[0], err)
        new = _memory_id(args[1], err)
        if old is None or new is None:
            return 2
        return _memories_supersede(store, old, new, out, err)
    finally:
        store.close()


def _memories_list(store: MemoryStore, out: IO[str]) -> int:
    rows = store.list_memories()
    if not rows:
        total = len(store.list_memories(include_superseded=True))
        out.write(
            f"bevel memories: no active memories. {total} superseded row(s) remain in "
            f"{store.db_path}; read one with 'bevel memories show <id>'.\n"
        )
        return 0
    out.write(f"{'id':>6}  {'type':<9}{'created':<17}{'scope':<34}content\n")
    for row in rows:
        # ``str:<N`` pads but never truncates, so anything wider than its column
        # runs into the next one. A real scope is a normalised remote URL, which
        # routinely is. Capped at 33, not 34: the column is 34 wide and the last
        # cell has to stay a space, or the table stops being a table.
        scope = row.scope if len(row.scope) <= 33 else f"{row.scope[:30]}..."
        content = row.content if len(row.content) <= 72 else f"{row.content[:71]}..."
        out.write(f"{row.id:>6}  {row.type!s:<9}{_stamp(row.created_at):<17}{scope:<34}{content}\n")
    hidden = len(store.list_memories(include_superseded=True)) - len(rows)
    note = f", {hidden} superseded" if hidden else ""
    out.write(f"\n  {len(rows)} active memory(ies){note} in {store.db_path}\n")
    return 0


def _memories_show(store: MemoryStore, memory_id: int, out: IO[str], err: IO[str]) -> int:
    row = store.get(memory_id)
    if row is None:
        err.write(f"bevel memories show: no memory with id {memory_id}.\n")
        return 2
    out.write(f"#{row.id}  {_stamp(row.created_at)}  [{row.type}]  {row.scope}\n")
    out.write(f"  {row.content}\n")
    if row.vector_model is not None:
        out.write(f"  vector: {row.vector_model}\n")
    if row.superseded_by is not None:
        out.write(f"  superseded by #{row.superseded_by}\n")
    return 0


def _memories_supersede(store: MemoryStore, old: int, new: int, out: IO[str], err: IO[str]) -> int:
    try:
        store.supersede(old, by=new)
    except ValueError as exc:
        err.write(f"bevel memories supersede: {exc}\n")
        return 2
    out.write(f"bevel memories: #{old} is superseded by #{new}. #{old} is no longer recalled.\n")
    return 0


def _resolve_session(
    store: SessionStore, named: str | None, err: IO[str], command: str
) -> SessionInfo | None:
    """The session a bare-or-named positional refers to, or ``None`` after writing why.

    Shared by ``resume`` and ``todos``, which resolve the argument identically:
    nothing named means the most recent session for this directory, and a prefix
    the store cannot resolve uniquely is the store's error to phrase.
    """
    if named is None:
        session_id = store.latest_session_id(Path.cwd())
        if session_id is None:
            err.write(
                f"bevel {command}: no stored session for {Path.cwd()}."
                f" Name one with 'bevel {command} <id>' (see 'bevel sessions').\n"
            )
            return None
        return store.resolve(session_id)
    try:
        return store.resolve(named)
    except ValueError as exc:
        err.write(f"bevel {command}: {exc}\n")
        return None


def _todos(named: str | None, out: IO[str], err: IO[str]) -> int:
    """Print the task list a session left behind.

    Read-only, and it does not create the database: on a fresh machine this says
    so and exits 0, the way ``bevel sessions`` does. Clearing a list is the
    model's job (it sends an empty list), not something the CLI offers -- a user
    with a spare ``clear`` would be one mistyped id away from destroying the
    plan they came to read.
    """
    from .config.loader import session_db_path  # noqa: PLC0415

    store = TodoStore(session_db_path())
    try:
        if not store.db_path.exists():
            out.write(f"bevel todos: no sessions have been stored yet ({store.db_path}).\n")
            return 0
        sessions = SessionStore(store.db_path)
        try:
            session = _resolve_session(sessions, named, err, "todos")
        finally:
            sessions.close()
        if session is None:
            return 2
        store.session_id = session.id
        todos = store.list_todos()
        if not todos:
            out.write(
                f"bevel todos: no todos stored for session {session.id}"
                f" ({_stamp(session.last_activity_at)}, {session.workspace}).\n"
            )
            return 0
        done = sum(1 for todo in todos if todo.status is TodoStatus.DONE)
        out.write(
            f"session {session.id}  {_stamp(session.last_activity_at)}  {session.workspace}\n"
        )
        for todo in todos:
            out.write(f"  [{_TODO_MARKS[todo.status]}] {todo.content}\n")
        out.write(f"\n  {done} of {len(todos)} done\n")
        return 0
    finally:
        store.close()


def _mcp(action: str | None, server: str | None, out: IO[str], err: IO[str]) -> int:
    """``bevel mcp list`` / ``bevel mcp resources <server>``.

    There is no ``start``/``stop``. A server's lifetime belongs to a run -- the
    TUI spawns them at the start of a turn sequence and kills the whole tree when
    it ends -- and a CLI that left a server running after the shell exited would
    be a process leak with a friendly name.
    """
    if action is None:
        err.write("bevel mcp: expected an action: list, resources.\n")
        return 2
    if action == "resources":
        return _mcp_resources(server, out, err)
    return _mcp_list(out, err)


def _mcp_list(out: IO[str], err: IO[str]) -> int:
    """Spawn every configured server, report, and tear it all down again.

    The servers are really started. A list command that read the config and
    printed it back would tell you your TOML is well-formed and nothing else,
    which is not the question -- the question is whether the server answers, and
    only asking it can say so.

    A server that will not start is reported and exits 0: telling the truth about
    a down server *is* this command succeeding. A config that cannot be read
    exits 2, because then nothing was reported faithfully.
    """
    from .config.loader import load  # noqa: PLC0415
    from .config.schema import ConfigError  # noqa: PLC0415

    try:
        config = load()
    except ConfigError as exc:
        err.write(f"bevel mcp list: {exc}\n")
        return 2

    if not config.mcp_servers:
        out.write(
            "bevel mcp list: no MCP servers configured.\n"
            "  Add a [mcp.servers.<name>] table to bevel.toml or ~/.config/bevel/config.toml;\n"
            "  each one declares exactly one of 'command' (a process bevel spawns) or 'url'\n"
            "  (a streamable-HTTP endpoint). See bevel.toml.example for every key.\n"
        )
        return 0

    reports = asyncio.run(_collect_reports(config.mcp_servers))
    out.write(f"{'server':<17}{'state':<10}{'status':<7}tools\n")
    for report in reports:
        state = "enabled" if report.enabled else "disabled"
        status = "up" if report.handshook else ("down" if report.enabled else "-")
        tools = ", ".join(name.split("__", 1)[-1] for name in report.tools) or "none"
        out.write(f"{report.name:<17}{state:<10}{status:<7}{tools}\n")
        out.write(f"{'':17}command: {' '.join(report.command)}\n")
        if report.error:
            out.write(f"{'':17}{report.error}\n")
        for refusal in report.refusals:
            out.write(f"{'':17}refused: {refusal.reason}\n")
    up = sum(1 for report in reports if report.handshook)
    out.write(f"\n  {up} of {len(reports)} server(s) answered\n")
    return 0


def _mcp_resources(server: str | None, out: IO[str], err: IO[str]) -> int:
    """``bevel mcp resources <server>``: the same call the model makes.

    It runs :class:`~bevel.mcp.resources.McpResourcesTool` rather than a second
    printing path, so a human and a model ask the server the same question the
    same way and cannot get different answers. Down, disabled, and empty are
    three different sentences because they are three different facts; a blank
    line for all three would send a user off to debug the wrong thing.

    Exits 2 when the server could not be reached and 0 when it answered, empty or
    not -- same rule as ``list``, where reporting a down server truthfully is the
    command succeeding, and a server with nothing to offer answered the question.
    """
    from .config.loader import load  # noqa: PLC0415
    from .config.schema import ConfigError  # noqa: PLC0415

    if server is None:
        err.write(
            "bevel mcp resources: which server? Pass its name: bevel mcp resources <server>.\n"
        )
        return 2

    try:
        config = load()
    except ConfigError as exc:
        err.write(f"bevel mcp resources: {exc}\n")
        return 2

    if not config.mcp_servers:
        err.write(
            "bevel mcp resources: no MCP servers configured, so there is no server to ask.\n"
            "  Add a [mcp.servers.<name>] table; see bevel.toml.example for every key.\n"
        )
        return 2

    content, error = asyncio.run(_list_one(config.mcp_servers, server))
    if error is not None:
        err.write(f"bevel mcp resources: {error}\n")
        return 2
    out.write(f"{content}\n")
    return 0


async def _list_one(servers: Mapping[str, McpServerConfig], name: str) -> tuple[str, str | None]:
    """Start, ask one server for its resources, and close everything again.

    One event loop, and the ``finally`` is the point: these are this process's
    own children and they must be gone when the command prints.
    """
    from .core.tools.base import ToolContext, call_tool  # noqa: PLC0415
    from .mcp.resources import McpResourcesTool  # noqa: PLC0415
    from .mcp.servers import start_servers  # noqa: PLC0415

    started = await start_servers(servers)
    try:
        tool = McpResourcesTool(started)
        result = await call_tool(tool, {"server": name}, ToolContext(workspace=Path.cwd()))
        return result.content, result.error
    finally:
        await started.aclose()


async def _collect_reports(servers: Mapping[str, McpServerConfig]) -> tuple[ServerReport, ...]:
    """Start, report, and close in one event loop.

    The ``finally`` is the reason this is a separate coroutine rather than two
    ``asyncio.run`` calls: the close has to happen on the loop that spawned the
    children, and it has to happen even if reading the reports raised.
    """
    from .mcp.servers import start_servers  # noqa: PLC0415

    started = await start_servers(servers)
    try:
        return started.reports
    finally:
        await started.aclose()


def _resume(named: str | None) -> int:
    """Reopen a session in the TUI: a named one, or the most recent for here."""
    from .config.loader import session_db_path  # noqa: PLC0415

    store = SessionStore(session_db_path())
    try:
        if not store.db_path.exists():
            sys.stderr.write(f"bevel resume: no sessions have been stored yet ({store.db_path}).\n")
            return 2
        session = _resolve_session(store, named, sys.stderr, "resume")
        if session is None:
            return 2
        session_id = session.id
    finally:
        store.close()

    # Imported here rather than at module scope, as with the TUI itself.
    from .tui.app import run_tui  # noqa: PLC0415

    return run_tui(resume=session_id)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911 - one exit per subcommand
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

    if args.command == "memories":
        return _memories(args.action, args.args, sys.stdout, sys.stderr)

    if args.command == "resume":
        return _resume(args.session)

    if args.command == "todos":
        return _todos(args.session, sys.stdout, sys.stderr)

    if args.command == "mcp":
        return _mcp(args.action, args.server, sys.stdout, sys.stderr)

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

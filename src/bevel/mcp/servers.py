"""Start the configured MCP servers and hand back their tools.

This is the only module in the package that owns a server's *lifecycle*.
:func:`~bevel.mcp.bridge.bridge_tools` deliberately does not spawn or close
anything, and neither :class:`~bevel.mcp.stdio.StdioTransport` nor
:class:`~bevel.mcp.http.HttpTransport` knows anything about configuration; the
seam between them is here, where a dict of
:class:`~bevel.config.schema.McpServerConfig` becomes a set of live servers and
the tools they contributed.

Which transport a server gets is decided by the one key its table carries:
``command`` means a child process, ``url`` means a streamable-HTTP endpoint. The
loader has already refused a table with both or neither, so nothing here
arbitrates -- and nothing below the report cares which one it got, which is why
:attr:`ServerReport.command` holds the URL for an HTTP server rather than the
dataclass growing a second field the CLI would have to branch on.

Two rules shape everything below.

**A server that cannot start must not break the run.** A user with four servers
configured and one typo'd path is not asking to be refused the other three, so a
failure is recorded in that server's :class:`ServerReport` and the loop moves on.
The report carries the transport's own message, which names the stage (spawn,
handshake, call) and quotes the tail of the server's stderr -- the only thing a
user can actually act on.

**Everything started must be gone when :meth:`McpServers.aclose` returns.**
stdio servers are children of this process, so a leak there is a leaked process
tree on the user's machine. Each transport is closed individually and closing is
safe to repeat, so a caller that closes in a ``finally`` and an outer cleanup
both survive. The HTTP transport's guarantee is the opposite one and just as
much a leak if broken: it ends its own session and closes its own connection
pool, and never touches a process it did not start.

Servers are started one after another rather than concurrently. Startup is not
the bottleneck a run has, and sequential spawning keeps the reports in config
order, keeps the log readable when two servers are noisy, and means a hung
server costs one timeout rather than making every timeout concurrent.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from bevel.config.schema import McpServerConfig

from .bridge import BridgedServer, BridgedTool, ToolRefusal, bridge_tools
from .http import HttpTransport
from .stdio import McpTransportError, StdioTransport

#: Either transport, whichever the config's one key selected. Named once because
#: it appears in four signatures below and a reader should not have to expand it
#: each time.
Transport = StdioTransport | HttpTransport

logger = logging.getLogger(__name__)

__all__ = ["McpServers", "ServerReport", "start_servers"]


@dataclass(frozen=True)
class ServerReport:
    """What happened to one configured server, as ``bevel mcp list`` shows it.

    A report exists for every configured server including the disabled and the
    dead, because a server that silently vanished from a list is indistinguishable
    from a typo in the table above it.

    ``pid`` is kept after the server has been closed so a caller -- and the tests
    -- can ask whether the child is really gone rather than hoping the code
    remembered to look beforehand.
    """

    name: str
    enabled: bool
    handshook: bool
    error: str | None
    tools: tuple[str, ...] = ()
    refusals: tuple[ToolRefusal, ...] = ()
    pid: int | None = None
    #: What to show for this server: its argv, or its URL for a streamable-HTTP
    #: one. Either way it is the thing a user compares against what they meant.
    #: Never the ``env`` values.
    command: tuple[str, ...] = ()


class McpServers:
    """The servers a caller started, and the only way to stop them.

    Built by :func:`start_servers`; there is no other way to get one, so a
    transport cannot exist here without having been started, and cannot be
    forgotten about either.
    """

    def __init__(
        self,
        transports: Sequence[tuple[Transport | None, ServerReport]],
        tools: Sequence[BridgedTool],
    ) -> None:
        self._transports = list(transports)
        # Kept apart from the transports so the reports survive ``aclose``: a
        # caller that has already torn everything down still has to be able to
        # say what it tore down, and a test can only ask "is the child gone?"
        # after the fact.
        self._reports = tuple(report for _transport, report in transports)
        self._tools = tuple(tools)

    @property
    def tools(self) -> tuple[BridgedTool, ...]:
        """The bridged tools, in config order, collisions already removed.

        Each satisfies :class:`~bevel.core.tools.base.Tool`, which is the only
        contract that matters where they are handed to ``Agent(tools=...)``.
        """
        return self._tools

    @property
    def reports(self) -> tuple[ServerReport, ...]:
        """One per configured server, in config order; still there after closing."""
        return self._reports

    async def aclose(self) -> None:
        """Close every transport, including the ones that failed part-way.

        Safe to call twice and safe to call when nothing started. Each close is
        guarded so one transport that refuses to die cannot strand the ones after
        it -- the whole point of this method is that nothing is left running.
        """
        pending, self._transports = self._transports, []
        for transport, report in pending:
            if transport is None:
                continue
            try:
                await transport.close()
            except Exception as exc:
                # Deliberately broad: a transport that refuses to die must not
                # strand the servers queued behind it.
                logger.warning("mcp server %r: closing it raised %s", report.name, exc)


async def start_servers(configs: Mapping[str, McpServerConfig]) -> McpServers:
    """Spawn, handshake, and bridge every enabled server in ``configs``.

    Never raises because of a server that failed: the result is a report per
    server plus whatever tools survived. The caller owns the processes from the
    moment this returns and must call :meth:`McpServers.aclose`.
    """
    started: list[tuple[Transport | None, ServerReport]] = []
    tools: list[BridgedTool] = []
    #: offered name -> the server that claimed it first.
    taken: dict[str, str] = {}

    for name, config in configs.items():
        if not config.enabled:
            started.append((None, _disabled_report(name, config)))
            continue
        transport = _build_transport(config)
        report, bridged = await _start_one(transport, name, config, taken)
        tools.extend(bridged)
        # The transport is kept even when its report says it never came up:
        # ``close`` is safe on a transport that got as far as constructing a
        # client or a spawn attempt, and dropping it here is how a half-started
        # child or a half-open connection pool gets forgotten about.
        started.append((transport, report))
    return McpServers(started, tools)


def _build_transport(config: McpServerConfig) -> Transport:
    """The transport the config's one key selects.

    ``url`` means HTTP and ``command`` means a child process; the loader has
    already refused a table with both or neither, so there is nothing to arbitrate
    here beyond picking the branch. Each transport gets the configured ``timeout``
    for every phase it has, exactly as stdio has always done -- an HTTP server
    that loads a big model at startup needs the same knob raised for the same
    reason a stdio one does.
    """
    if config.url is not None:
        return HttpTransport(
            config.url,
            connect_timeout=config.timeout,
            handshake_timeout=config.timeout,
            call_timeout=config.timeout,
        )
    if config.command is None:  # pragma: no cover - the loader refuses this
        raise McpTransportError("this server declares neither command nor url")
    return StdioTransport(
        list(_argv(config)),
        cwd=config.cwd,
        env=_environment(config),
        spawn_timeout=config.timeout,
        handshake_timeout=config.timeout,
        call_timeout=config.timeout,
    )


def _argv(config: McpServerConfig) -> tuple[str, ...]:
    """What to show a user about this server: its argv, or its URL.

    One field in the report rather than two, because ``bevel mcp list`` prints
    one line per server and "command: https://…" reads honestly. Splitting it
    would mean the CLI has to know which transport a server was.
    """
    if config.url is not None:
        return (config.url,)
    return (config.command or "", *config.args)


def _environment(config: McpServerConfig) -> dict[str, str]:
    """The parent environment plus the configured additions.

    ``create_subprocess_exec`` *replaces* the environment when handed a mapping,
    so passing ``config.env`` straight through would strip ``PATH`` and the spawn
    would fail with a ``FileNotFoundError`` blaming a command that exists. Merging
    is the deliberate choice: it is the behaviour that works, and restricting a
    server's environment is the sandbox milestone's job, not this layer's.
    """
    return {**os.environ, **config.env}


def _disabled_report(name: str, config: McpServerConfig) -> ServerReport:
    """A disabled server is reported, not skipped.

    A server that quietly disappeared from ``bevel mcp list`` would look exactly
    like a mistyped table name, so the row is printed with ``enabled`` false and
    no process is ever created for it.
    """
    return ServerReport(
        name=name,
        enabled=False,
        handshook=False,
        error=None,
        command=_argv(config),
    )


async def _start_one(
    transport: Transport,
    name: str,
    config: McpServerConfig,
    taken: dict[str, str],
) -> tuple[ServerReport, tuple[BridgedTool, ...]]:
    """Bring one server up, or explain why it is not up. Never raises.

    Returns the report and the tools it earned, kept apart because the report is
    what a user reads and the tools are what the run is handed; a report with
    tools in it but no way to call them would be the worst of both.
    """
    try:
        await transport.start()
        bridged = await bridge_tools(transport, name)
    except McpTransportError as exc:
        # ``start`` closes the transport when it fails, and ``close`` is
        # idempotent, so there is nothing left to clean up on this path.
        logger.warning("mcp server %r did not start: %s", name, exc)
        return (
            ServerReport(
                name=name,
                enabled=True,
                handshook=False,
                error=f"{name}: {exc}",
                command=_argv(config),
            ),
            (),
        )

    tools, refusals = _drop_cross_server_collisions(name, bridged, taken)
    return (
        ServerReport(
            name=name,
            enabled=True,
            handshook=True,
            error=None,
            tools=tuple(tool.spec.name for tool in tools),
            refusals=refusals,
            pid=transport.pid,
            command=_argv(config),
        ),
        tools,
    )


def _drop_cross_server_collisions(
    name: str, bridged: BridgedServer, taken: dict[str, str]
) -> tuple[tuple[BridgedTool, ...], tuple[ToolRefusal, ...]]:
    """Keep the first claim on an offered name and refuse the rest, by name.

    :func:`~bevel.mcp.bridge.bridge_tools` dedupes *within* one server, so two
    servers named ``my-server`` and ``my_server`` both hand back
    ``my_server__echo``. ``Agent`` keys tools by name and would silently keep
    whichever arrived last, which means the ordering of a config file decides a
    capability the model can see. Refusing the later one and naming both servers
    is the only honest option -- "a name collided" without saying which two is not
    something a user can act on.
    """
    kept: list[BridgedTool] = []
    refusals = list(bridged.refusals)
    for tool in bridged.tools:
        offered = tool.spec.name
        previous = taken.get(offered)
        if previous is not None:
            reason = (
                f"{tool.remote_name}: offered as {offered!r}, the same name already offered by "
                f"mcp server {previous!r}; refusing the one from {name!r} rather than letting "
                "config order decide which one the model gets"
            )
            refusals.append(ToolRefusal(tool.remote_name, reason))
            logger.warning(
                "mcp server %r: refusing %r -- already claimed by %r", name, offered, previous
            )
            continue
        taken[offered] = name
        kept.append(tool)
    return tuple(kept), tuple(refusals)

"""The m10d oracle: configured servers started from a config file and bridged.

Everything here goes through :func:`bevel.mcp.servers.start_servers`, which is
the only place in the package that owns a server's *lifecycle* --
:func:`~bevel.mcp.bridge.bridge_tools` deliberately does not, and a test that
spawned a transport by hand would be testing m10c again rather than the wiring
this slice adds.

The oracle is ``tests/fixtures/mcp_server.py`` spawned for real by argv from a
config the test wrote. There is no fake transport and no monkeypatched
:class:`~bevel.mcp.stdio.StdioTransport` except where a test is specifically
pinning that a disabled server was never constructed at all -- and there the
replacement records constructor calls rather than answering RPCs, which is a
statement about the code path, not a substitute for the wire.

Two behaviours are the reason this module exists and are asserted first: a
server that cannot start must not break the run, and everything that was
started must be gone when :meth:`McpServers.aclose` returns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from bevel.mcp.servers import McpServers, start_servers

from bevel.config.schema import McpServerConfig
from bevel.core.agent import Agent
from bevel.core.policy import allow_all_policy
from bevel.core.tools import registry
from bevel.core.tools.base import ToolContext, call_tool
from bevel.mcp import servers as servers_module
from bevel.providers.mock import MockProvider, ScriptedTurn

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"

#: Copied from ``test_mcp_bridge.py`` rather than imported, so that adding a
#: built-in fails this file loudly instead of quietly making "the built-ins
#: survived bridging" vacuous.
BUILTIN_TOOLS = ["read", "grep", "glob", "recall", "todo", "write", "edit", "remember", "shell"]


def fixture_config(*flags: str, **overrides: Any) -> McpServerConfig:
    """A server config that spawns the fixture, with the given extra flags."""
    fields: dict[str, Any] = {
        "command": sys.executable,
        "args": (str(FIXTURE), *flags),
    }
    fields.update(overrides)
    return McpServerConfig(**fields)


_STILL_ACTIVE = 259


def _process_alive(pid: int | None) -> bool:  # noqa: PLR0911 - one return per OS probe
    """Liveness without adding a dependency; copied from ``test_mcp_stdio.py``.

    Copied rather than imported because there is no ``tests`` package to import
    from, and a leaked child is the failure this file has to be able to see.
    """
    if pid is None:
        return False
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def _started(*flags: str, name: str = "fixture", **overrides: Any) -> McpServers:
    return await start_servers({name: fixture_config(*flags, **overrides)})


async def _call(servers: McpServers, name: str, **arguments: Any):
    tool = next(tool for tool in servers.tools if tool.spec.name == name)
    return await call_tool(tool, arguments, ToolContext(workspace=Path.cwd()))


# ---------------------------------------------------------------------------
# The round trip from a config object to a tool the model is offered.
# ---------------------------------------------------------------------------


async def test_a_configured_server_is_spawned_handshaked_and_bridged() -> None:
    servers = await _started()
    try:
        report = servers.reports[0]
        assert report.name == "fixture"
        assert report.handshook is True
        assert report.error is None
        assert "fixture__echo" in report.tools
    finally:
        await servers.aclose()


async def test_a_bridged_tool_from_config_actually_executes() -> None:
    """Handshaking proves the pipes opened; only calling one proves the tool works."""
    servers = await _started()
    try:
        result = await _call(servers, "fixture__echo", text="from config")
        assert result.ok, result.content
        assert "from config" in result.content
    finally:
        await servers.aclose()


async def test_a_run_gets_the_bridged_tool_and_every_builtin() -> None:
    """The whole point of the slice: bridged tools reach a real ``Agent`` run.

    ``Agent(tools=...)`` *replaces* the built-ins rather than extending them, so
    the assertion has to be on both halves. A wiring bug that passed only the
    bridged tools would leave the model unable to read a file, and would still
    have offered ``fixture__echo``.
    """
    provider = MockProvider(turns=[ScriptedTurn(content="done")])
    servers = await _started()
    try:
        agent = Agent(
            provider,
            Path.cwd(),
            tools=[*registry.tools(), *servers.tools],
            policy=allow_all_policy,
        )
        result = await agent.run("hello")
        assert result.ok, result.error
        offered = provider.requests[0]["tools"]
        assert isinstance(offered, list)
        assert "fixture__echo" in offered
        assert offered[: len(BUILTIN_TOOLS)] == BUILTIN_TOOLS
    finally:
        await servers.aclose()


async def test_starting_servers_does_not_touch_the_registry() -> None:
    """Re-pinned from this slice's angle: the config path must not register either."""
    before = registry.specs()
    assert [spec.name for spec in before] == BUILTIN_TOOLS
    servers = await _started()
    try:
        assert registry.specs() == before
        assert registry.get("fixture__echo") is None
    finally:
        await servers.aclose()


async def test_no_configured_servers_yields_nothing_and_closes_cleanly() -> None:
    servers = await start_servers({})
    assert servers.tools == ()
    assert servers.reports == ()
    await servers.aclose()


# ---------------------------------------------------------------------------
# Lifecycle: everything started must die.
# ---------------------------------------------------------------------------


async def test_aclose_kills_every_child_it_started() -> None:
    """A leaked child is a leaked process tree on the user's machine.

    The liveness check is copied from ``test_mcp_stdio.py`` rather than imported,
    as the rest of this package's helpers are: there is no ``tests`` package to
    import from. A report keeps the pid it saw, so this can ask the question
    after the fact rather than hoping it remembered to look beforehand.
    """
    servers = await start_servers(
        {
            "one": fixture_config(),
            "two": fixture_config("--stderr-lines", "2"),
        }
    )
    pids = [report.pid for report in servers.reports]
    assert len(pids) == 2
    assert all(pid is not None for pid in pids)
    assert all(_process_alive(pid) for pid in pids)

    await servers.aclose()
    assert all(not _process_alive(pid) for pid in pids)


async def test_aclose_is_safe_to_call_twice() -> None:
    """The TUI closes in a ``finally``; a second close from an outer cleanup must not raise."""
    servers = await _started()
    await servers.aclose()
    await servers.aclose()


async def test_a_server_that_never_started_is_still_closed_without_error() -> None:
    servers = await start_servers({"broken": fixture_config(command="definitely-not-a-program")})
    assert servers.reports[0].handshook is False
    await servers.aclose()


# ---------------------------------------------------------------------------
# One bad server must not break the run.
# ---------------------------------------------------------------------------


async def test_a_server_that_cannot_spawn_does_not_stop_the_others() -> None:
    """The failure the user actually hits: one typo'd path in a config of four.

    A raise here would mean bevel refuses to start because a server the user
    forgot to delete is broken, so the failure has to be data, not an exception.
    """
    servers = await start_servers(
        {
            "broken": fixture_config(command="definitely-not-a-program"),
            "good": fixture_config(),
        }
    )
    try:
        assert [report.name for report in servers.reports] == ["broken", "good"]
        assert [report.handshook for report in servers.reports] == [False, True]
        broken = servers.reports[0]
        assert broken.error is not None
        assert "broken" in broken.error, "the message must name which server failed"
        assert "definitely-not-a-program" in broken.error
        assert [tool.spec.name for tool in servers.tools] == [
            "good__echo",
            "good__boom",
            "good__methods",
            "good__read_note",
        ]
    finally:
        await servers.aclose()


async def test_a_server_that_fails_the_handshake_reports_its_stderr() -> None:
    """``stderr_tail`` is what turns "the server hung" into a diagnosis.

    ``--stderr-lines`` writes at startup and ``--hang initialize`` then goes
    silent, so the only thing a user has to act on is what the server printed.
    """
    servers = await start_servers(
        {"noisy": fixture_config("--stderr-lines", "2", "--hang", "initialize", timeout=3.0)}
    )
    try:
        report = servers.reports[0]
        assert report.handshook is False
        assert report.error is not None
        assert "noisy" in report.error
        assert "handshake timed out" in report.error, "the message must name the stage"
        assert "fixture stderr line 0" in report.error, "the server's own stderr must be quoted"
    finally:
        await servers.aclose()


async def test_a_broken_server_leaves_no_tools_behind() -> None:
    servers = await start_servers({"broken": fixture_config(command="definitely-not-a-program")})
    try:
        assert servers.tools == ()
    finally:
        await servers.aclose()


# ---------------------------------------------------------------------------
# Each config key has to reach the child process, or it is decoration.
# ---------------------------------------------------------------------------


async def test_env_merges_with_the_parent_environment() -> None:
    """``env`` adds to the inherited environment rather than replacing it.

    ``create_subprocess_exec`` *replaces* the environment when handed a mapping,
    so a config that set one variable would strip ``PATH`` and the spawn would
    fail with a ``FileNotFoundError`` blaming a command that exists. The fixture
    reports what it actually sees, which is the only proof that counts here.
    """
    servers = await _started(
        "--report-env", "BEVEL_MCP_PROBE", env={"BEVEL_MCP_PROBE": "configured"}
    )
    try:
        result = await _call(servers, "fixture__echo", text="ignored")
        assert result.content.strip() == "BEVEL_MCP_PROBE=configured"
    finally:
        await servers.aclose()


async def test_env_does_not_strip_the_path_the_server_needs_to_run() -> None:
    """The other half of the merge: the child still has a working environment."""
    servers = await _started("--report-env", "PATH", env={"BEVEL_MCP_PROBE": "x"})
    try:
        result = await _call(servers, "fixture__echo", text="ignored")
        reported = result.content.split("=", 1)[1]
        assert reported != "<unset>", "a replaced environment would leave the child no PATH"
        assert reported == os.environ["PATH"]
    finally:
        await servers.aclose()


async def test_cwd_is_where_the_server_runs(tmp_path: Path) -> None:
    expected = tmp_path / "elsewhere"
    expected.mkdir()
    servers = await _started("--report-cwd", cwd=expected)
    try:
        result = await _call(servers, "fixture__echo", text="ignored")
        assert Path(result.content.strip()) == expected
    finally:
        await servers.aclose()


async def test_timeout_reaches_the_transport_and_names_the_stage() -> None:
    """One ``timeout`` key, applied to the stage that actually stalls.

    ``--hang initialize`` goes silent after the child is alive, so this is the
    handshake stage rather than the spawn stage, and the message has to say so.
    """
    servers = await _started("--hang", "initialize", timeout=2.0)
    try:
        assert servers.reports[0].error is not None
        assert "handshake timed out after 2.0s" in servers.reports[0].error
    finally:
        await servers.aclose()


async def test_a_disabled_server_is_never_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    """``enabled = false`` must mean no process, not a process that is ignored.

    The transport class is replaced with a recorder, which is a statement about
    which code path ran. Nothing answers RPCs here, so nothing below can pass by
    talking to a fake.
    """
    constructed: list[list[str]] = []

    class NeverSpawned:
        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            constructed.append(list(argv))

    monkeypatch.setattr(servers_module, "StdioTransport", NeverSpawned)
    servers = await start_servers({"off": fixture_config(enabled=False)})
    assert constructed == [], "a disabled server was spawned anyway"
    report = servers.reports[0]
    assert report.enabled is False
    assert report.handshook is False
    assert report.tools == ()
    await servers.aclose()


# ---------------------------------------------------------------------------
# Names the model is offered: collisions across servers.
# ---------------------------------------------------------------------------


async def test_two_servers_colliding_on_one_offered_name_offer_it_once() -> None:
    """``my-server`` and ``my_server`` both sanitise to ``my_server``.

    :func:`~bevel.mcp.bridge.bridge_tools` dedupes *within* a server, so both
    hand back ``my_server__echo``. ``Agent`` keys tools by name and would
    silently keep one, so the collision is detected here and the loser is
    refused by name -- naming both servers, because "a name collided" without
    saying which two is not actionable.
    """
    servers = await start_servers({"my-server": fixture_config(), "my_server": fixture_config()})
    try:
        offered = [tool.spec.name for tool in servers.tools]
        assert offered.count("my_server__echo") == 1
        reasons = [refusal.reason for report in servers.reports for refusal in report.refusals]
        assert any(
            "my-server" in reason and "my_server" in reason and "echo" in reason
            for reason in reasons
        ), f"no refusal named both servers and the tool: {reasons}"
    finally:
        await servers.aclose()


async def test_a_bridged_name_can_never_shadow_a_builtin() -> None:
    """An invariant across two modules that do not know about each other.

    The ``__`` separator makes a collision structurally impossible today, and
    nothing enforces that it stays that way: the registry pins its names, the
    bridge pins its separator, and neither imports the other. Asserted rather
    than assumed, because a shadowed built-in would be a silent downgrade of
    what the model can do.
    """
    servers = await _started("--adversarial-tools")
    try:
        offered = {tool.spec.name for tool in servers.tools}
        assert offered.isdisjoint(BUILTIN_TOOLS)
    finally:
        await servers.aclose()


async def test_a_refused_descriptor_is_reported_and_costs_only_itself() -> None:
    """The bridge already refuses a broken descriptor; this checks it survives to the report."""
    servers = await _started("--adversarial-tools")
    try:
        assert servers.reports[0].refusals, "the adversarial descriptors were not refused"
        assert any(refusal.reason for refusal in servers.reports[0].refusals), (
            "a refusal must carry a reason the user can read"
        )
        assert servers.tools, "refusals must not take the working tools with them"
    finally:
        await servers.aclose()

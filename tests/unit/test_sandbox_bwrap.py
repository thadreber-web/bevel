"""Tests for the m12a bwrap backend: the prefix, the probe, the self-test gate.

The oracle for the kernel tests at the bottom is a real child process denied
by a real kernel -- the same shape as m10b's fixture and m11b's gh tests.
They are Linux-only and *fail* rather than skip on Linux when bwrap cannot
run, because a silently-skipped kernel test is the silent fake this project
forbids. On Windows they skip with a named reason; the ubuntu CI lane and
WSL2 are where they run.

Everything above the kernel section is platform-independent: argv building,
probe parsing, and the degrade ladder are exercised by monkeypatching so the
same assertions hold on both CI lanes.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from bevel.core.agent import Agent
from bevel.core.sandbox import bwrap
from bevel.core.sandbox.base import EnforcementLevel, SandboxStatus
from bevel.core.sandbox.selftest import (
    SelfTestReport,
    _run_child,
    connect_child_argv,
    run_selftest,
    write_child_argv,
)
from bevel.core.tools.base import ToolContext, call_tool
from bevel.core.tools.shell import ShellTool, _spawn_argv
from bevel.providers.mock import MockProvider

IS_LINUX = sys.platform == "linux"

BWRAP = "/usr/bin/bwrap"
WORKSPACE = Path("/home/dev/project")
# The two argv positions that go through ``str(Path(...))`` in build_prefix
# normalise their separators on Windows; comparing against the normalised
# form keeps these structure tests honest on both CI lanes. On Linux these
# equal the ``/usr/bin/bwrap`` and ``/home/dev/project`` literals verbatim.
BWRAP_S = str(Path(BWRAP))
WORKSPACE_S = str(WORKSPACE)


def _probe_result(
    *,
    exit_code: int | None = 0,
    stdout: str = "ok\n",
    stderr: str = "",
    timed_out: bool = False,
    spawn_error: str | None = None,
) -> bwrap.ProbeResult:
    return bwrap.ProbeResult(
        command=(BWRAP, *bwrap.PROBE_ARGV),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        spawn_error=spawn_error,
    )


class TestEnforcementLevel:
    def test_exactly_two_states(self) -> None:
        """No slots for the deleted Landlock/seccomp/AppContainer/T2 ladder."""
        assert set(EnforcementLevel) == {EnforcementLevel.POLICY_ONLY, EnforcementLevel.BWRAP}

    def test_badge_strings(self) -> None:
        assert EnforcementLevel.POLICY_ONLY.value == "policy-only"
        assert EnforcementLevel.BWRAP.value == "bwrap"


class TestBuildPrefix:
    def test_prefix_is_the_planned_argv(self) -> None:
        """Every flag is pinned by name: the kernel enforces exactly this list."""
        prefix = bwrap.build_prefix(Path(BWRAP), WORKSPACE)
        assert prefix == (
            BWRAP_S,
            "--ro-bind",
            "/",
            "/",
            "--bind",
            WORKSPACE_S,
            WORKSPACE_S,
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--unshare-net",
            "--unshare-user",
            "--die-with-parent",
            "--",
        )

    def test_workspace_is_bound_writable_at_its_own_path(self) -> None:
        prefix = bwrap.build_prefix(Path(BWRAP), WORKSPACE)
        bind = prefix.index("--bind")
        assert prefix[bind + 1 : bind + 3] == (WORKSPACE_S, WORKSPACE_S)

    def test_root_is_read_only(self) -> None:
        prefix = bwrap.build_prefix(Path(BWRAP), WORKSPACE)
        ro = prefix.index("--ro-bind")
        assert prefix[ro + 1 : ro + 3] == ("/", "/")
        # The ro-bind of / comes before the rw-bind of the workspace, or the
        # workspace bind would be shadowed by the read-only root.
        assert ro < prefix.index("--bind")

    def test_argv_terminates_with_the_separator(self) -> None:
        """'--' ends bwrap's own options; everything after is the tool's argv."""
        assert bwrap.build_prefix(Path(BWRAP), WORKSPACE)[-1] == "--"

    def test_no_cpu_or_memory_flags(self) -> None:
        """No caps in this slice: bwrap has none and setrlimit is out of scope.

        The module docstring says the timeout + tree-kill are the backstop;
        this test pins that no one quietly added a cap that does not exist.
        """
        prefix = " ".join(bwrap.build_prefix(Path(BWRAP), WORKSPACE))
        for absent in ("--rlimit", "--cap", "--cpu", "--memory"):
            assert absent not in prefix


class TestResolveBwrap:
    def test_resolves_to_a_real_path_not_a_path_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The AppArmor grant is by path: a copy earlier on PATH is not covered."""
        real = tmp_path / "bin" / "bwrap"
        real.parent.mkdir()
        real.write_text("#!/bin/sh\n")
        monkeypatch.setattr(bwrap.shutil, "which", lambda _name: str(real))
        resolved = bwrap.resolve_bwrap()
        assert resolved is not None
        assert resolved == real.resolve()
        assert resolved.is_absolute()

    def test_absent_bwrap_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bwrap.shutil, "which", lambda _name: None)
        assert bwrap.resolve_bwrap() is None


class TestRunProbe:
    """The probe RUNS bwrap. shutil.which alone is the forbidden shortcut."""

    def test_probe_command_is_the_documented_one(self) -> None:
        assert bwrap.PROBE_ARGV == (
            "--ro-bind",
            "/",
            "/",
            "--unshare-user",
            "--unshare-net",
            "--",
            "/bin/echo",
            "ok",
        )

    def test_probe_passes_on_exit_zero_and_expected_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_run(argv: list[str], **kwargs: Any) -> Any:
            seen["argv"] = argv
            seen["shell"] = kwargs.get("shell", False)
            return _Completed(0, "ok\n", "")

        monkeypatch.setattr(bwrap.subprocess, "run", fake_run)
        result = bwrap.run_probe(Path(BWRAP))
        assert result.passed is True
        assert seen["argv"] == [BWRAP_S, *bwrap.PROBE_ARGV]
        assert seen["shell"] is False  # never shell=True

    def test_expected_stdout_is_required_not_just_exit_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bwrap that exits 0 without printing 'ok' did not run the probe."""
        monkeypatch.setattr(
            bwrap.subprocess, "run", lambda _argv, **_kw: _Completed(0, "something else", "")
        )
        assert bwrap.run_probe(Path(BWRAP)).passed is False

    def test_uid_map_failure_is_a_lying_install_not_a_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact Ubuntu 24.04 failure: installed, on PATH, and broken."""
        monkeypatch.setattr(
            bwrap.subprocess,
            "run",
            lambda _argv, **_kw: _Completed(1, "", "bwrap: setting up uid map: Permission denied"),
        )
        result = bwrap.run_probe(Path(BWRAP))
        assert result.passed is False
        assert result.exit_code == 1
        assert "uid map" in result.stderr
        fixes = " | ".join(result.fixes)
        assert "sudo apt install uidmap" in fixes
        assert "/etc/apparmor.d/bwrap" in fixes

    def test_spawn_error_is_recorded_as_an_error_not_silence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(argv: list[str], **kwargs: Any) -> Any:
            raise OSError("exec format error")

        monkeypatch.setattr(bwrap.subprocess, "run", boom)
        result = bwrap.run_probe(Path(BWRAP))
        assert result.passed is False
        assert result.spawn_error is not None
        assert "exec format error" in result.spawn_error

    def test_timeout_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def hang(argv: list[str], **kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 10))

        monkeypatch.setattr(bwrap.subprocess, "run", hang)
        result = bwrap.run_probe(Path(BWRAP))
        assert result.passed is False
        assert result.timed_out is True


class _Completed:
    def __init__(self, code: int, out: str, err: str) -> None:
        self.returncode = code
        self.stdout = out
        self.stderr = err


class TestDetect:
    """The degrade ladder: probe -> self-test -> level, with a loud reason."""

    def _patch_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bwrap, "_CURRENT_PLATFORM", "linux")
        monkeypatch.setattr(bwrap, "resolve_bwrap", lambda: Path(BWRAP))

    async def test_probe_pass_and_selftest_pass_is_bwrap(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_linux(monkeypatch)
        monkeypatch.setattr(bwrap, "run_probe", lambda _path: _probe_result())
        monkeypatch.setattr(bwrap, "run_selftest", _ok_selftest)
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.BWRAP
        assert status.degrade is None
        assert status.prefix[0] == BWRAP_S  # resolved path, not the bare name

    async def test_bwrap_absent_is_policy_only_naming_the_install(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bwrap, "_CURRENT_PLATFORM", "linux")
        monkeypatch.setattr(bwrap, "resolve_bwrap", lambda: None)
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.POLICY_ONLY
        assert status.degrade is not None
        assert "sudo apt install bubblewrap" in " | ".join(status.degrade.fixes)

    async def test_probe_fail_is_policy_only_with_verbatim_command_exit_and_stderr(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_linux(monkeypatch)
        monkeypatch.setattr(
            bwrap,
            "run_probe",
            lambda _path: _probe_result(
                exit_code=1, stdout="", stderr="bwrap: setting up uid map: Permission denied"
            ),
        )
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.POLICY_ONLY
        assert status.degrade is not None
        assert status.degrade.command == f"{BWRAP} {' '.join(bwrap.PROBE_ARGV)}"
        assert status.degrade.exit_code == 1
        assert "uid map" in status.degrade.stderr
        assert any("uidmap" in fix for fix in status.degrade.fixes)

    async def test_probe_lies_selftest_catches_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Probe said ok, the self-test's write succeeded: the probe lied.

        This is the case a probe-only design cannot see -- and the reason the
        self-test exists. The drop to POLICY_ONLY must name the self-test.
        """
        self._patch_linux(monkeypatch)
        monkeypatch.setattr(bwrap, "run_probe", lambda _path: _probe_result())
        monkeypatch.setattr(bwrap, "run_selftest", _failed_selftest)
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.POLICY_ONLY
        assert status.degrade is not None
        assert "self-test" in status.degrade.reason
        assert "probe" in status.degrade.reason

    async def test_windows_gets_no_new_behavior(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """m12b owns Windows. No probe runs, no degrade report, no badge."""
        monkeypatch.setattr(bwrap, "_CURRENT_PLATFORM", "win32")

        def never() -> Path:
            raise AssertionError("the probe must not run on Windows")

        monkeypatch.setattr(bwrap, "resolve_bwrap", never)
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.POLICY_ONLY
        assert status.degrade is None

    async def test_badge_text(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._patch_linux(monkeypatch)
        monkeypatch.setattr(bwrap, "run_probe", lambda _path: _probe_result())
        monkeypatch.setattr(bwrap, "run_selftest", _ok_selftest)
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.badge == "sandbox: bwrap"

        monkeypatch.setattr(bwrap, "resolve_bwrap", lambda: None)
        degraded = await bwrap.detect_sandbox(tmp_path)
        assert degraded.badge == "sandbox: policy-only"

        monkeypatch.setattr(bwrap, "_CURRENT_PLATFORM", "win32")
        untouched = await bwrap.detect_sandbox(tmp_path)
        assert untouched.badge is None  # Windows renders nothing until m12b


async def _ok_selftest(prefix: tuple[str, ...], ws: Path) -> SelfTestReport:
    return SelfTestReport(
        ok=True, write_denied=True, write_detail="EROFS", net_denied=True, net_detail="EPERM"
    )


async def _failed_selftest(prefix: tuple[str, ...], ws: Path) -> SelfTestReport:
    return SelfTestReport(
        ok=False,
        write_denied=False,
        write_detail="touch exited 0 and the file exists",
        net_denied=True,
        net_detail="EPERM",
    )


class TestShellThroughPrefix:
    """The shell tool spawns through the prefix when the backend is active."""

    def test_the_prefix_is_prepended_to_the_model_argv(self) -> None:
        prefix = (BWRAP, "--ro-bind", "/", "/", "--")
        assert _spawn_argv(prefix, ["ls", "-la"]) == [*prefix, "ls", "-la"]

    def test_no_prefix_means_the_argv_is_untouched(self) -> None:
        """POLICY_ONLY must not rewrite the argv at all -- same spawn as today."""
        assert _spawn_argv((), ["ls", "-la"]) == ["ls", "-la"]


class TestAgentWiring:
    def test_agent_passes_the_prefix_to_the_tool_context(self, tmp_path: Path) -> None:
        status = SandboxStatus(
            level=EnforcementLevel.BWRAP,
            prefix=(BWRAP, "--"),
            probe_attempted=True,
            degrade=None,
        )
        agent = Agent(MockProvider(turns=[]), tmp_path, sandbox=status)
        assert agent.ctx.sandbox_prefix == (BWRAP, "--")

    def test_agent_without_a_sandbox_spawns_unprefixed(self, tmp_path: Path) -> None:
        """A directly-constructed Agent (tests, integration) keeps today's spawn."""
        agent = Agent(MockProvider(turns=[]), tmp_path)
        assert agent.ctx.sandbox_prefix == ()


# ---------------------------------------------------------------------------
# Kernel tests: a real child, denied by a real kernel.
#
# These spawn bwrap for real. They skip ONLY off Linux (named reason) and run
# on the ubuntu CI lane (which installs bubblewrap + uidmap) and under WSL2.
# On Linux without a working bwrap they FAIL -- a kernel test that silently
# skips is the silent fake this project forbids.
# ---------------------------------------------------------------------------

pytestmark_kernel = pytest.mark.skipif(
    not IS_LINUX,
    reason="kernel-enforced denial requires a Linux kernel with bwrap; "
    "m12a runs these on the ubuntu CI lane and under WSL2",
)


@pytest.fixture
def real_prefix(tmp_path: Path) -> tuple[str, ...]:
    path = bwrap.resolve_bwrap()
    if path is None:
        pytest.fail(
            "bwrap is not installed on this Linux box. The kernel tests are the "
            "m12a oracle and must not silently skip: sudo apt install bubblewrap uidmap"
        )
    result = bwrap.run_probe(path)
    if not result.passed:
        pytest.fail(
            f"bwrap is installed but the probe fails (exit {result.exit_code}): "
            f"{result.stderr}\nfixes: {'; '.join(result.fixes)}"
        )
    return bwrap.build_prefix(path, tmp_path)


@pytestmark_kernel
class TestKernelDenials:
    """The oracle: the KERNEL refuses, not bevel's Python.

    Each denial has a control test proving the same child *succeeds* without
    the prefix -- otherwise the denial test could pass for a reason that has
    nothing to do with the sandbox (an unroutable address, a missing binary).
    """

    async def test_write_outside_the_workspace_is_denied(
        self, real_prefix: tuple[str, ...], tmp_path: Path
    ) -> None:
        target = Path(f"/tmp/bevel-m12a-kernel-write-{uuid.uuid4().hex}.txt")
        argv = write_child_argv(real_prefix, target)
        code, _out, err = await _run_child(argv, str(tmp_path), 15.0)
        assert code not in (0, None), f"the kernel did not deny the write: {err!r}"
        created = target.exists()  # noqa: ASYNC240 - a local temp file, once
        assert not created, "the file was created despite the non-zero exit"

    async def test_write_control_without_the_prefix_succeeds(self, tmp_path: Path) -> None:
        """If this passes denied too, the denial test above proves nothing."""
        target = Path(f"/tmp/bevel-m12a-kernel-write-control-{uuid.uuid4().hex}.txt")
        argv = write_child_argv((), target)
        code, _out, err = await _run_child(argv, str(tmp_path), 15.0)
        assert code == 0, f"the unsandboxed control write failed: {err!r}"
        assert target.exists()  # noqa: ASYNC240 - a local temp file, once
        target.unlink(missing_ok=True)  # noqa: ASYNC240 - a local temp file, once

    async def test_outbound_connect_is_denied(
        self, real_prefix: tuple[str, ...], tmp_path: Path
    ) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            argv = connect_child_argv(real_prefix, port)
            code, out, err = await _run_child(argv, str(tmp_path), 15.0)
            assert code not in (0, None), (
                "the kernel did not deny the connect -- the child reached a real "
                f"listener: out={out!r} err={err!r}"
            )
        finally:
            listener.close()

    async def test_connect_control_without_the_prefix_succeeds(self, tmp_path: Path) -> None:
        """The listener is real; unsandboxed, the child must connect."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            argv = connect_child_argv((), port)
            code, out, err = await _run_child(argv, str(tmp_path), 15.0)
            assert code == 0, f"the unsandboxed control connect failed: out={out!r} err={err!r}"
        finally:
            listener.close()

    async def test_selftest_passes_under_a_real_bwrap(
        self, real_prefix: tuple[str, ...], tmp_path: Path
    ) -> None:
        report = await run_selftest(real_prefix, tmp_path)
        assert report.ok is True, (
            f"write_denied={report.write_denied} ({report.write_detail}) "
            f"net_denied={report.net_denied} ({report.net_detail})"
        )

    async def test_detect_reaches_bwrap_on_a_working_machine(self, tmp_path: Path) -> None:
        """End to end on a real kernel: probe + self-test -> BWRAP."""
        status = await bwrap.detect_sandbox(tmp_path)
        assert status.level is EnforcementLevel.BWRAP, (
            f"expected BWRAP on a machine where the probe passes; degrade={status.degrade}"
        )

    async def test_shell_tool_write_outside_workspace_is_denied(
        self, real_prefix: tuple[str, ...], tmp_path: Path
    ) -> None:
        """The shipped path: the shell tool spawning through the prefix.

        The model's argv asks for a write outside the workspace; the prefix
        is what refuses it, and the refusal comes from the kernel -- the
        shell tool's own Python never looks at the command's paths.
        """
        target = Path(f"/tmp/bevel-m12a-shell-write-{uuid.uuid4().hex}.txt")
        ctx = ToolContext(tmp_path, sandbox_prefix=real_prefix)
        code = f"open({str(target)!r}, 'w').write('x'); print('WROTE')"
        result = await call_tool(ShellTool(), {"command": [sys.executable, "-c", code]}, ctx)
        assert result.ok is True, "a denied write is a reported exit code, not a tool crash"
        assert "[exit 0]" not in result.content, (
            f"the kernel did not deny the shell tool's out-of-workspace write: {result.content!r}"
        )
        created = target.exists()  # noqa: ASYNC240 - a local temp file, once
        assert not created

    async def test_shell_tool_write_inside_workspace_still_works(
        self, real_prefix: tuple[str, ...], tmp_path: Path
    ) -> None:
        """The bind is real: the workspace stays writable, or the tool is useless."""
        ctx = ToolContext(tmp_path, sandbox_prefix=real_prefix)
        code = "open('inside.txt', 'w').write('x'); print('WROTE')"
        result = await call_tool(ShellTool(), {"command": [sys.executable, "-c", code]}, ctx)
        assert "[exit 0]" in result.content, result.content
        written = (tmp_path / "inside.txt").exists()
        assert written

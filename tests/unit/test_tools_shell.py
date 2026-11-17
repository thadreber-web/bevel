"""Tests for the ``shell`` tool.

The two behaviours that make this tool safe to hand to a model are the ones
that are hardest to fake: a timeout that actually stops the command, and a kill
that takes the whole process tree rather than the parent (an orphaned grandchild
keeps the pipe open and the next call hangs). Everything runs through
``sys.executable`` so the tests do not depend on what happens to be installed on
the machine or the CI image.
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from bevel.core.tools.base import RiskTier, ToolContext, ToolResult, call_tool
from bevel.core.tools.shell import ShellTool

IS_WINDOWS = sys.platform == "win32"
PROBE_PORT = 45_318


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


async def shell(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(ShellTool(), arguments, ctx)


def py(code: str) -> list[str]:
    """An argv list that runs ``code`` with this interpreter."""
    return [sys.executable, "-c", code]


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259


def process_alive(pid: int) -> bool:
    """Best-effort liveness check without adding a dependency."""
    if IS_WINDOWS:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
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


def port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class TestDeclaration:
    def test_shell_is_exec_unsafe_and_not_parallel_safe(self) -> None:
        tool = ShellTool()
        assert tool.risk_tier is RiskTier.EXEC_UNSAFE
        assert tool.is_parallel_safe is False

    def test_only_command_is_required(self) -> None:
        assert ShellTool().spec.parameters["required"] == ["command"]

    def test_command_is_an_argv_list_not_a_string(self) -> None:
        items = ShellTool().spec.parameters["properties"]["command"]["items"]
        assert items["type"] == "string"
        assert ShellTool().spec.parameters["properties"]["command"]["type"] == "array"


class TestRunning:
    async def test_captures_stdout_and_reports_the_exit_code(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print('hello')"))
        assert result.ok is True
        assert "hello" in result.content
        assert "exit 0" in result.content

    async def test_captures_stdout_and_stderr_separately(self, ctx: ToolContext) -> None:
        result = await shell(
            ctx, command=py("import sys; print('out'); print('err', file=sys.stderr)")
        )
        assert "out" in result.content
        assert "err" in result.content
        assert "stderr" in result.content

    async def test_a_nonzero_exit_is_a_result_not_a_failure(self, ctx: ToolContext) -> None:
        """`pytest` exits 1 when tests fail; the model still needs the output."""
        result = await shell(ctx, command=py("raise SystemExit(3)"))
        assert result.ok is True
        assert "exit 3" in result.content

    async def test_runs_in_the_workspace_by_default(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("import os; print(os.getcwd())"))
        assert result.ok is True
        assert str(ctx.workspace) in result.content.replace("\\\\", "\\")

    async def test_cwd_selects_a_subdirectory(self, ctx: ToolContext) -> None:
        (ctx.workspace / "sub").mkdir()
        result = await shell(ctx, command=py("import os; print(os.getcwd())"), cwd="sub")
        assert str(ctx.workspace / "sub") in result.content.replace("\\\\", "\\")

    async def test_cwd_outside_the_workspace_is_refused(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print(1)"), cwd="../")
        assert result.failed
        assert "outside the workspace" in (result.error or "")

    async def test_cwd_must_be_a_directory(self, ctx: ToolContext) -> None:
        (ctx.workspace / "file.txt").write_text("x")
        result = await shell(ctx, command=py("print(1)"), cwd="file.txt")
        assert result.failed

    async def test_a_missing_working_directory_is_a_failure(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print(1)"), cwd="nope")
        assert result.failed

    async def test_a_missing_program_is_a_failure_naming_it(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=["definitely-not-a-real-program-xyz"])
        assert result.failed
        assert "definitely-not-a-real-program-xyz" in (result.error or "")

    async def test_stdin_is_closed_so_readers_do_not_hang(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("import sys; print(repr(sys.stdin.read()))"))
        assert result.ok is True
        assert "''" in result.content

    async def test_each_stream_keeps_its_own_order(self, ctx: ToolContext) -> None:
        result = await shell(
            ctx,
            command=py(
                "import sys; print('one'); print('two'); print('a', file=sys.stderr); "
                "print('b', file=sys.stderr)"
            ),
        )
        assert result.content.index("one") < result.content.index("two")
        assert result.content.index("a") < result.content.index("b")


class TestArgumentGuards:
    async def test_an_empty_argv_is_refused(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=[])
        assert result.failed

    async def test_a_command_string_is_rejected_by_the_schema(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command="echo hi")
        assert result.failed
        assert "command" in (result.error or "")

    async def test_a_zero_timeout_is_refused(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print(1)"), timeout=0)
        assert result.failed

    async def test_an_absurd_timeout_is_capped(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print(1)"), timeout=10**6)
        assert result.failed
        assert "at most" in (result.error or "")

    async def test_paths_outside_the_workspace_are_refused(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print(1)"), cwd="../../")
        assert result.failed
        assert "outside the workspace" in (result.error or "")


class TestOutputLimits:
    async def test_large_output_is_capped_and_marked(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("print('x' * 100000)"), max_output_bytes=2000)
        assert result.ok is True
        assert len(result.content) < 20_000
        assert "truncated" in result.content.lower()

    async def test_truncation_keeps_the_head_not_the_tail(self, ctx: ToolContext) -> None:
        result = await shell(
            ctx,
            command=py("print('HEAD'); print('x' * 100000); print('TAIL')"),
            max_output_bytes=2000,
        )
        assert "HEAD" in result.content
        assert "TAIL" not in result.content


class TestTimeouts:
    async def test_a_timeout_is_a_failure_that_names_the_limit(self, ctx: ToolContext) -> None:
        result = await shell(ctx, command=py("import time; time.sleep(30)"), timeout=1)
        assert result.failed
        assert "timed out" in (result.error or "").lower()
        assert "1" in (result.error or "")

    async def test_a_timeout_returns_promptly_rather_than_waiting_for_the_child(
        self, ctx: ToolContext
    ) -> None:
        start = time.monotonic()
        await shell(ctx, command=py("import time; time.sleep(60)"), timeout=1)
        assert time.monotonic() - start < 15

    async def test_output_produced_before_a_timeout_is_still_returned(
        self, ctx: ToolContext
    ) -> None:
        result = await shell(
            ctx,
            command=py("import sys, time; print('before', flush=True); time.sleep(30)"),
            timeout=1,
        )
        assert result.failed
        assert "before" in result.content

    async def test_a_timeout_kills_the_whole_process_tree(self, ctx: ToolContext) -> None:
        """The grandchild must die with the parent, not orphan and hold the pipe."""
        code = (
            "import subprocess, sys, time; "
            f"child = subprocess.Popen([sys.executable, '-c', "
            f"'import socket, time; s=socket.socket(); s.setsockopt(1,2,1); "
            f's.bind(("127.0.0.1", {PROBE_PORT})); s.listen(1); print("up", flush=True); '
            "time.sleep(120)']); "
            "print(child.pid, flush=True); time.sleep(120)"
        )
        result = await shell(ctx, command=py(code), timeout=3)
        assert result.failed
        pid_lines = [line for line in result.content.splitlines() if line.strip().isdigit()]
        assert pid_lines, f"grandchild never reported its pid: {result.content!r}"
        pid = int(pid_lines[0])
        assert pid > 0
        for _ in range(50):
            if not port_listening(PROBE_PORT):
                break
            await asyncio.sleep(0.2)
        assert not port_listening(PROBE_PORT), "grandchild survived the tree kill"
        assert not process_alive(pid), "grandchild process is still alive"

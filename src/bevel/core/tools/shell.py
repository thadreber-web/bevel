"""The ``shell`` tool: run a command, with a timeout that really stops it.

Two rules from AGENTS.md shape everything here.

**Never ``shell=True``.** ``command`` is an argv array and stays one all the way
to :func:`asyncio.create_subprocess_exec`. Routing through ``cmd.exe`` would
inherit its quoting rules, which is a command-injection surface dressed up as
convenience -- and the model is the one typing the string.

**Kill trees, not parents.** A timeout that only kills the direct child leaves
grandchildren alive and holding the pipes open, so the next call hangs. Windows
gets a Job Object marked ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; POSIX gets its
own process group and ``killpg``. Both are kernel mechanisms, not a helper
process walking the tree in user space.

The tier is ``EXEC_UNSAFE`` and nothing better: whether a particular argv is
safe is a decision about the *command*, made by a rule, and rules are m05.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

#: ``sys.platform`` behind an opaque ``str``. mypy special-cases comparisons
#: against ``sys.platform`` and, with ``warn_unreachable`` on, marks the other
#: OS's branches dead -- which breaks whichever typecheck lane is not the
#: host's. Both lanes must typecheck the code as written.
_CURRENT_PLATFORM: Final[str] = sys.platform

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_TIMEOUT_SECONDS: Final[float] = 600.0
DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 32_000
MAX_OUTPUT_BYTES_CAP: Final[int] = 1_000_000
#: Bytes taken from a pipe per read.
_CHUNK_BYTES: Final[int] = 65_536
#: How long to let the pipes drain after a kill, so output produced *before*
#: the timeout is not thrown away with the process.
_DRAIN_GRACE_SECONDS: Final[float] = 2.0

# --- Windows Job Object constants -------------------------------------------
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: Final[int] = 0x0000_2000
_JobObjectExtendedLimitInformation: Final[int] = 9
_PROCESS_TERMINATE: Final[int] = 0x0001
_PROCESS_SET_QUOTA: Final[int] = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000


class _IO_COUNTERS(ctypes.Structure):  # noqa: N801 - mirrors the Win32 struct name
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):  # noqa: N801 - Win32 struct name
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(  # noqa: N801 - Win32 struct name
    ctypes.Structure
):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@dataclass
class _Stream:
    """One captured pipe: the kept bytes, and whether more existed."""

    chunks: list[bytes] = field(default_factory=list)
    kept: int = 0
    total: int = 0
    cap: int = DEFAULT_MAX_OUTPUT_BYTES

    @property
    def truncated(self) -> bool:
        return self.total > self.kept

    def accept(self, chunk: bytes) -> None:
        """Record a chunk, keeping only the first ``cap`` bytes of the stream.

        Reading continues past the cap (discarding) rather than stopping,
        because a pipe nobody drains keeps the writer blocked and the child
        unwaitable.
        """
        self.total += len(chunk)
        if self.kept >= self.cap:
            return
        room = self.cap - self.kept
        self.chunks.append(chunk[:room])
        self.kept += min(len(chunk), room)

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


class ShellTool:
    """Run a command in the workspace and capture its output."""

    spec: Final[ToolSpec] = ToolSpec(
        name="shell",
        description=(
            "Run a command and return its stdout, stderr and exit code. `command` is "
            "an argv array, never a shell string: put each argument in its own "
            "element. The command runs in the workspace with a timeout, and a "
            "timeout kills the whole process tree. A non-zero exit code is reported, "
            "not treated as a tool failure."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "description": "The program and its arguments, one array element each.",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory, relative to the workspace (default: the workspace)."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Seconds before the command is killed (default {DEFAULT_TIMEOUT_SECONDS:g})."
                    ),
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS,
                },
                "max_output_bytes": {
                    "type": "integer",
                    "description": (
                        f"Bytes kept from each output stream (default {DEFAULT_MAX_OUTPUT_BYTES})."
                    ),
                    "minimum": 1,
                    "maximum": MAX_OUTPUT_BYTES_CAP,
                    "default": DEFAULT_MAX_OUTPUT_BYTES,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.EXEC_UNSAFE
    is_parallel_safe: Final[bool] = False

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        argv = [str(item) for item in arguments["command"]]
        if not argv[0].strip():
            raise ToolError("'command' must name a program to run")

        cwd = self._working_directory(arguments, ctx)
        timeout = float(arguments.get("timeout", DEFAULT_TIMEOUT_SECONDS))
        cap = min(
            int(arguments.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)), MAX_OUTPUT_BYTES_CAP
        )

        proc = await _spawn(argv, os.fspath(cwd))
        killer = _TreeKiller(proc.pid)
        stdout = _Stream(cap=cap)
        stderr = _Stream(cap=cap)
        timed_out = False
        try:
            timed_out = await _run(proc, stdout, stderr, timeout, killer)
        finally:
            killer.close()
            _close(proc)

        body = _render(stdout, stderr, proc.returncode)
        if timed_out:
            return ToolResult(
                ok=False,
                content=body,
                error=(
                    f"shell: timed out after {timeout:g}s running {argv[0]!r}; "
                    "the process tree was killed"
                ),
            )
        return ToolResult(ok=True, content=body)

    def _working_directory(
        self, arguments: Mapping[str, Any], ctx: ToolContext
    ) -> os.PathLike[str]:
        if "cwd" not in arguments:
            return ctx.workspace
        cwd = ctx.require_within_workspace(str(arguments["cwd"]))
        if not cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {ctx.display(cwd)}")
        return cwd


async def _spawn(argv: list[str], cwd: str) -> asyncio.subprocess.Process:
    """Start the child, in its own process group where that exists.

    ``start_new_session`` is POSIX-only -- ``subprocess.Popen`` raises
    ``ValueError`` for it on Windows -- so it is passed conditionally. Windows
    gets its tree kill from the Job Object instead.
    """
    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
    }
    if _CURRENT_PLATFORM != "win32":
        kwargs["start_new_session"] = True
    try:
        return await asyncio.create_subprocess_exec(*argv, **kwargs)
    except FileNotFoundError as exc:
        raise ToolError(f"program not found: {argv[0]!r}") from exc
    except PermissionError as exc:
        raise ToolError(f"permission denied running {argv[0]!r}: {exc}") from exc
    except OSError as exc:
        raise ToolError(f"could not run {argv[0]!r}: {type(exc).__name__}: {exc}") from exc


async def _run(
    proc: asyncio.subprocess.Process,
    stdout: _Stream,
    stderr: _Stream,
    timeout_seconds: float,
    killer: _TreeKiller,
) -> bool:
    """Wait for the command; kill its tree on timeout. Returns whether it timed out."""
    tasks = {
        asyncio.ensure_future(_drain(proc.stdout, stdout)),
        asyncio.ensure_future(_drain(proc.stderr, stderr)),
        asyncio.ensure_future(proc.wait()),
    }
    _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    if not pending:
        return False

    killer.kill()
    # The pipes only reach EOF once the whole tree is dead, so give the kernel a
    # moment to close them; anything already written is still reported.
    await asyncio.wait(pending, timeout=_DRAIN_GRACE_SECONDS)
    for task in pending:
        task.cancel()
    return True


async def _drain(stream: asyncio.StreamReader | None, sink: _Stream) -> None:
    """Read a pipe to EOF."""
    if stream is None:  # pragma: no cover - both pipes are always created
        return
    while True:
        chunk = await stream.read(_CHUNK_BYTES)
        if not chunk:
            return
        sink.accept(chunk)


def _close(proc: asyncio.subprocess.Process) -> None:
    """Release the transport. Only legal once the child has been reaped.

    ``Process.close`` exists at runtime but typeshed does not declare it, so it
    is looked up rather than called -- the same trick used for the POSIX-only
    members of :mod:`os` below, and for the same reason: neither typecheck lane
    may be told a member exists that its own platform does not have.
    """
    if proc.returncode is None:
        return
    close = getattr(proc, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _kill_signal() -> int | None:
    """The signal number to send when a tree must die now.

    ``SIGKILL`` is what the POSIX branch wants. It does not exist on Windows,
    where ``os.kill`` accepts any number and terminates the process with it as
    the exit code, so ``SIGTERM`` is an equally fatal substitute there.
    """
    for name in ("SIGKILL", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is not None:
            return int(number)
    return None  # pragma: no cover - every platform has at least one


def _render(stdout: _Stream, stderr: _Stream, exit_code: int | None) -> str:
    parts: list[str] = []
    if stdout.text().strip():
        parts.append(stdout.text().rstrip("\n"))
    if stdout.truncated:
        parts.append(f"[stdout truncated at {stdout.cap} bytes]")
    if stderr.text().strip():
        parts.append(f"[stderr]\n{stderr.text().rstrip(chr(10))}")
    if stderr.truncated:
        parts.append(f"[stderr truncated at {stderr.cap} bytes]")
    parts.append(f"[exit {exit_code if exit_code is not None else 'unknown'}]")
    return "\n".join(parts)


class _TreeKiller:
    """Kills a child *and everything it spawned*.

    Windows: the child is assigned to a Job Object marked
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Closing the handle makes the kernel
    terminate every process in the job, however deeply nested -- including
    descendants created after the assignment, which is what makes this a tree
    kill rather than a single-process kill.

    POSIX: the child was started with ``start_new_session``, so it leads its own
    process group and ``killpg`` reaches its descendants too.

    Both paths degrade to a plain per-process kill rather than raising: a
    timeout that kills something is better than one that kills nothing.
    """

    def __init__(self, pid: int) -> None:
        self._pid = pid
        self._kernel32: ctypes.CDLL | None = None
        self._job: int | None = None
        if _CURRENT_PLATFORM == "win32":
            self._assign_to_job()

    def _assign_to_job(self) -> None:
        try:
            kernel32 = ctypes.CDLL("kernel32")
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel32.SetInformationJobObject.restype = ctypes.c_bool
            kernel32.SetInformationJobObject.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            kernel32.AssignProcessToJobObject.restype = ctypes.c_bool
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]
            kernel32.CloseHandle.restype = ctypes.c_bool
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job,
                _JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                kernel32.CloseHandle(job)
                return
            handle = kernel32.OpenProcess(
                _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                self._pid,
            )
            if not handle:
                kernel32.CloseHandle(job)
                return
            try:
                if not kernel32.AssignProcessToJobObject(job, handle):
                    kernel32.CloseHandle(job)
                    return
            finally:
                kernel32.CloseHandle(handle)
        except OSError:
            # No kernel32, or a kernel that refused the job. _kill falls back.
            return
        self._kernel32 = kernel32
        self._job = int(job)

    def kill(self) -> None:
        """Terminate the tree. Safe to call more than once."""
        if self._job is not None and self._kernel32 is not None:
            kernel32, job = self._kernel32, self._job
            self._job = None
            # Closing the job handle *is* the kill: KILL_ON_JOB_CLOSE takes every
            # member, including grandchildren created after assignment.
            with contextlib.suppress(OSError):
                kernel32.CloseHandle(job)
            return
        if _CURRENT_PLATFORM == "win32":
            self._taskkill()
        else:
            self._killpg()
        self._kill_directly()

    def close(self) -> None:
        """Release the job handle on the normal path, where nothing must die."""
        if self._job is not None and self._kernel32 is not None:
            with contextlib.suppress(OSError):
                self._kernel32.CloseHandle(self._job)
            self._job = None

    def _taskkill(self) -> None:
        # Fallback only, for when the Job Object could not be created. taskkill
        # walks the tree in user space, so it can race a fast-exiting child.
        # The variable is spelled `SystemRoot`, not `SYSTEMROOT`.
        root = os.environ.get("SystemRoot", r"C:\Windows")  # noqa: SIM112
        taskkill = Path(root) / "System32" / "taskkill.exe"
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(  # noqa: S603 - fixed argv, no shell, no user text
                [os.fspath(taskkill), "/T", "/F", "/PID", str(self._pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )

    def _killpg(self) -> None:
        """Signal the child's whole process group.

        ``os.killpg`` is POSIX-only, so it is fetched by name for the same
        typeshed reason as ``Process.close`` above.
        """
        killpg = getattr(os, "killpg", None)
        number = _kill_signal()
        if killpg is None or number is None:
            return
        with contextlib.suppress(OSError):
            killpg(self._pid, number)

    def _kill_directly(self) -> None:
        number = _kill_signal()
        if number is None:
            return
        with contextlib.suppress(OSError):
            os.kill(self._pid, number)


__all__ = ["ShellTool"]

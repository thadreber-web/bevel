"""Killing a child process *and everything it spawned*.

Two places need this: the ``shell`` tool (a command that outlives its timeout)
and the checkpoint module (a ``git`` that wedges on a lock file). Both were
written against the same two rules from AGENTS.md -- always a timeout, always a
process-tree kill -- and a tree kill is not something to reimplement twice, so
it lives here.

**Windows:** the child is assigned to a Job Object marked
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Closing the handle makes the kernel
terminate every process in the job, however deeply nested -- including
descendants created *after* the assignment, which is what makes this a tree
kill rather than a single-process kill.

**POSIX:** the child must be started with ``start_new_session`` so it leads its
own process group; ``killpg`` then reaches its descendants too.

Both paths degrade to a plain per-process kill rather than raising. A timeout
that kills something is better than one that kills nothing.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Final

#: ``sys.platform`` behind an opaque ``str``. mypy special-cases comparisons
#: against ``sys.platform`` and, with ``warn_unreachable`` on, marks the other
#: OS's branches dead -- which breaks whichever typecheck lane is not the
#: host's. Both lanes must typecheck the code as written.
_CURRENT_PLATFORM: Final[str] = sys.platform

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


def uses_new_session() -> bool:
    """Whether a child must be started with ``start_new_session``.

    ``subprocess.Popen`` raises ``ValueError`` for that keyword on Windows, so
    the caller cannot pass it unconditionally; Windows gets its tree kill from
    the Job Object instead.
    """
    return _CURRENT_PLATFORM != "win32"


def kill_signal() -> int | None:
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


def close_process(proc: object) -> None:
    """Release the transport. Only legal once the child has been reaped.

    ``Process.close`` exists at runtime but typeshed does not declare it, so it
    is looked up rather than called -- the same trick used for the POSIX-only
    members of :mod:`os` below, and for the same reason: neither typecheck lane
    may be told a member exists that its own platform does not have.
    """
    if getattr(proc, "returncode", None) is None:
        return
    close = getattr(proc, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


class TreeKiller:
    """Kills a child *and everything it spawned*.

    Construct one per child, call :meth:`kill` to end the tree, and
    :meth:`close` on the normal path to release the job handle without killing
    anything. Both are safe to call more than once.
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
            # No kernel32, or a kernel that refused the job. kill() falls back.
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
        number = kill_signal()
        if killpg is None or number is None:
            return
        with contextlib.suppress(OSError):
            killpg(self._pid, number)

    def _kill_directly(self) -> None:
        number = kill_signal()
        if number is None:
            return
        with contextlib.suppress(OSError):
            os.kill(self._pid, number)


__all__ = ["TreeKiller", "close_process", "kill_signal", "uses_new_session"]

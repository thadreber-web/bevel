"""The self-test: a real child, denied by a real kernel.

This module is the m12a oracle. The probe in :mod:`bevel.core.sandbox.bwrap`
is a smoke test -- it proves bwrap *starts*. Only this module proves the
sandbox *contains*: it spawns real children through the bwrap prefix and
requires the **kernel** to refuse them. A Python-side refusal (bevel checking
the path before spawning) proves nothing about the sandbox -- that path is
already covered by ``tests/unit/test_tools_base.py`` and would pass with zero
kernel enforcement in place.

The two denials, each with a control test that proves the same child
*succeeds* without the prefix (``tests/unit/test_sandbox_bwrap.py``):

``write_child_argv``
    attempts ``open(target, "w")`` on a path outside the workspace. Under
    ``--ro-bind / /`` the kernel refuses (EROFS/EACCES) and the child exits
    non-zero. If it exits 0, the bind is not what we think it is.

``connect_child_argv``
    attempts a TCP ``connect`` to a listener that exists in the *parent*
    network namespace. Under ``--unshare-net`` the child has no route to it
    and the kernel refuses. If it connects, the namespace was never unshared.

If either denial fails, :func:`run_selftest` reports ``ok=False`` and
``detect_sandbox`` drops to POLICY_ONLY loudly -- the probe passed and the
self-test caught it lying, which is exactly the case a probe-only design
cannot see.

What this module cannot prove, stated plainly
---------------------------------------------
* **It tests the prefix, not every future spawn.** The self-test runs the
  same prefix the shell tool will use, but it cannot guarantee that some
  future code path forgets to apply it. That guarantee lives in review and
  in the shell tool's tests, not here.
* **It proves denial of these two operations only.** Filesystem writes
  outside the workspace and outbound connects are what the tier claims;
  ptrace, /proc abuse beyond the mount, and resource exhaustion are not
  probed -- and the tier makes no claim about them (see the no-caps note in
  :mod:`bevel.core.sandbox.bwrap`).
* **A kernel that denies today can deny differently tomorrow.** The level
  is re-verified at every process start, not cached to disk; a machine that
  upgrades its kernel gets a fresh verdict on the next run rather than a
  remembered one.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bevel.core._process import TreeKiller, close_process, uses_new_session

__all__ = [
    "SelfTestReport",
    "connect_child_argv",
    "run_selftest",
    "write_child_argv",
]

#: How long one self-test child may run. These children do nothing but one
#: syscall; anything slower than this is wedged and counts as not-denied
#: (fail closed: a child we cannot read a verdict from is a child the kernel
#: did not verifiably refuse).
CHILD_TIMEOUT_SECONDS: Final[float] = 15.0

#: The outbound connect target port is supplied by the caller (a real
#: listener in the parent namespace); this is the child's own connect budget.
_CONNECT_TIMEOUT_SECONDS: Final[float] = 5.0


@dataclass(frozen=True)
class SelfTestReport:
    """The verdict of the two kernel denials.

    ``ok`` is the only field callers branch on; the details exist so the
    loud degrade can quote *which* denial failed and what the child actually
    said, rather than reporting "the self-test failed".
    """

    ok: bool
    write_denied: bool
    write_detail: str
    net_denied: bool
    net_detail: str

    @property
    def detail(self) -> str:
        """A one-line account of the failures, for the degrade report."""
        parts = []
        if not self.write_denied:
            parts.append(f"out-of-workspace write was NOT denied: {self.write_detail}")
        if not self.net_denied:
            parts.append(f"outbound connect was NOT denied: {self.net_detail}")
        return "; ".join(parts) if parts else "both denials confirmed by the kernel"


def write_child_argv(prefix: tuple[str, ...], target: Path) -> list[str]:
    """The full argv for a child that tries to create ``target`` (outside the workspace).

    Exposed (not private) so the kernel tests can run the *same* child the
    self-test runs -- against the prefix, and against no prefix as the
    control. One child, two assertions, no drift between them.
    """
    code = f"open({str(target)!r}, 'w').write('bevel-m12a-selftest')"
    return [*prefix, sys.executable, "-c", code]


def connect_child_argv(prefix: tuple[str, ...], port: int) -> list[str]:
    """The full argv for a child that tries to TCP-connect to 127.0.0.1:``port``.

    The listener is real and lives in the parent namespace, so an
    unsandboxed child must connect (the control) and a sandboxed one must be
    refused by the kernel.
    """
    code = (
        "import socket, sys\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        f"s.settimeout({_CONNECT_TIMEOUT_SECONDS})\n"
        "try:\n"
        f"    s.connect(('127.0.0.1', {port}))\n"
        "except OSError as exc:\n"
        "    print(f'{type(exc).__name__}: {exc}', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('connected')\n"
    )
    return [*prefix, sys.executable, "-c", code]


async def _run_child(
    argv: list[str], cwd: str, timeout_seconds: float
) -> tuple[int | None, str, str]:
    """Spawn one self-test child and return (exit code, stdout, stderr).

    Same spawn discipline as every other bevel child: argv list, never
    ``shell=True``, own process group on POSIX, and a :class:`TreeKiller` so
    a wedged child cannot outlive the self-test. A child that times out
    reports ``None`` as its exit code -- callers treat that as not-denied.
    """
    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
    }
    if uses_new_session():
        kwargs["start_new_session"] = True
    proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
    killer = TreeKiller(proc.pid)
    try:
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            killer.kill()
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(proc.communicate(), timeout=2.0)
            return None, "", f"the self-test child timed out after {timeout_seconds:g}s"
        return (
            proc.returncode,
            raw_out.decode("utf-8", "replace"),
            raw_err.decode("utf-8", "replace"),
        )
    finally:
        killer.close()
        close_process(proc)


def _outside_workspace_target() -> Path:
    """A path outside the workspace that does not exist yet.

    ``/tmp`` is the documented target -- under ``--ro-bind / /`` it is the
    read-only root, and it is where the control test can prove the same
    child succeeds unsandboxed. The name is unique per call so two runs
    never race on one file.
    """
    return Path("/tmp") / f"bevel-sandbox-selftest-{os.getpid()}-{uuid.uuid4().hex[:8]}.txt"  # noqa: S108


async def run_selftest(prefix: tuple[str, ...], workspace: Path) -> SelfTestReport:
    """Spawn real children through ``prefix`` and require the kernel to deny them.

    Returns ``ok=False`` (never raises) when either operation was *not*
    denied or could not be judged -- fail closed: an unreadable verdict is
    not a verified boundary, and POLICY_ONLY with a loud reason is always
    the safe answer to ambiguity.
    """
    target = _outside_workspace_target()
    try:
        argv = write_child_argv(prefix, target)
        code, _out, err = await _run_child(argv, str(workspace), CHILD_TIMEOUT_SECONDS)
        write_denied = code is not None and code != 0
        write_detail = (
            f"exit {code}, stderr: {err.strip()[:300]}"
            if write_denied
            else f"exit {code}, the file exists: {target.exists()}"
        )
    except OSError as exc:
        write_denied = False
        write_detail = f"could not spawn the write child: {type(exc).__name__}: {exc}"
    finally:
        with contextlib.suppress(OSError):
            target.unlink(missing_ok=True)

    listener: socket.socket | None = None
    try:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        code, out, err = await _run_child(
            connect_child_argv(prefix, port), str(workspace), CHILD_TIMEOUT_SECONDS
        )
        net_denied = code is not None and code != 0
        net_detail = (
            f"exit {code}, stderr: {err.strip()[:300]}"
            if net_denied
            else f"exit {code}, stdout: {out.strip()[:300]}"
        )
    except OSError as exc:
        net_denied = False
        net_detail = f"could not set up the connect probe: {type(exc).__name__}: {exc}"
    finally:
        if listener is not None:
            listener.close()

    return SelfTestReport(
        ok=write_denied and net_denied,
        write_denied=write_denied,
        write_detail=write_detail,
        net_denied=net_denied,
        net_detail=net_detail,
    )

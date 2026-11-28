"""The bwrap backend: bevel builds the argv, the kernel does the enforcing.

The division of labour is the one ``gh`` has in m11b: bevel never contains
anything itself, it constructs an argv prefix and ``bwrap`` (a user-installed
distro package, joining ``git`` and ``gh``) makes the kernel enforce it. The
prefix the shell tool spawns through when this backend is active::

    <resolved bwrap> --ro-bind / / --bind <workspace> <workspace>
        --dev /dev --proc /proc --unshare-net --unshare-user
        --die-with-parent -- <argv>

Every flag is pinned by name in ``tests/unit/test_sandbox_bwrap.py``.

The decisions, already made
---------------------------
**No CPU/memory caps.** ``bwrap`` provides none (it is a mount/namespace
tool, not a resource controller) and ``setrlimit`` in a ``preexec_fn`` is
out of scope for this slice. The backstop against a runaway command is the
shell tool's timeout plus the :class:`~bevel.core._process.TreeKiller`,
which is what it was before m12 and has not changed. The tier name
``BWRAP`` claims filesystem and network isolation only, and this paragraph
is where that claim is written down rather than implied.

**Resolved path, always.** The user's AppArmor grant on Ubuntu 24.04 is by
path (``/usr/bin/bwrap`` gets the ``userns`` permission); a second copy
earlier on ``PATH`` is not covered by it and fails with a uid-map refusal
that looks like a broken install. So bevel resolves ``shutil.which`` to a
real path once and spawns that path, never the bare name.

**The probe RUNS bwrap.** ``shutil.which`` alone is the forbidden shortcut:
on Ubuntu 24.04 *installed is not working*. Missing ``uidmap`` helpers and
a missing or stale AppArmor profile both fail with ``setting up uid map:
Permission denied`` on a machine where bwrap is "installed". The probe is
``bwrap --ro-bind / / --unshare-user --unshare-net -- /bin/echo ok`` and
requires exit 0 *and* the expected stdout. A probe that could not execute
at all (spawn error) and a probe that ran and said no are distinct cases --
both unavailable, only the first is an error -- matching the
``Capability.error`` rule in :mod:`bevel._capability`.

**The probe is not trusted.** It is a smoke test, not a proof. The proof is
:mod:`bevel.core.sandbox.selftest`: a real child denied a forbidden write
and a forbidden connect *by the kernel*. If the probe passed and the
self-test's write succeeded, the probe lied and the level drops to
POLICY_ONLY with a degrade report that says so.

**No reused container.** Each tool call spawns its own bwrap; there is no
long-lived sandbox to invalidate when the workspace changes. **No
``--untrusted-repo`` flag** -- that belonged to the deleted tier ladder, and
declaring a flag this milestone does not read is the failure mode AGENTS.md
names. **Windows gets no new behavior**: m12a does not probe there and
renders no badge; m12b makes the Windows decision (T0 + a WSL2 pointer).

Assumptions pinned by tests (stated here rather than asked about)
-----------------------------------------------------------------
* The workspace is bind-mounted at its own path (``--bind W W``), so a
  command's absolute paths mean the same thing inside and outside the
  sandbox. Pinned by ``test_workspace_is_bound_writable_at_its_own_path``.
* ``--dev``/``--proc`` are included because a tool that cannot see ``/dev``
  or ``/proc`` is broken, not sandboxed; ``--unshare-pid`` is *not* included
  because it is not in the plan's flag list and hiding PIDs buys nothing the
  tree-kill does not already handle. Pinned by ``test_prefix_is_the_planned_argv``.
* The probe's expected stdout is the exact string ``ok`` after stripping, so
  a bwrap that exits 0 without running ``/bin/echo`` fails the probe. Pinned
  by ``test_expected_stdout_is_required_not_just_exit_zero``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from bevel.core.sandbox.base import DegradeReport, EnforcementLevel, SandboxStatus
from bevel.core.sandbox.selftest import SelfTestReport, run_selftest

__all__ = [
    "PROBE_ARGV",
    "ProbeResult",
    "build_prefix",
    "detect_sandbox",
    "resolve_bwrap",
    "run_probe",
]

#: ``sys.platform`` behind an opaque ``str``, the same trick
#: :mod:`bevel._capability` uses: mypy special-cases comparisons against
#: ``sys.platform`` and, with ``warn_unreachable`` on, marks the other OS's
#: branches dead -- which breaks whichever typecheck lane is not the host's.
_CURRENT_PLATFORM: Final[str] = sys.platform

#: The documented probe: a minimal sandbox that must be able to run echo.
#: Pinned verbatim by the tests -- this string is what ``bevel doctor``
#: prints when it degrades, and what a bug report quotes.
PROBE_ARGV: Final[tuple[str, ...]] = (
    "--ro-bind",
    "/",
    "/",
    "--unshare-user",
    "--unshare-net",
    "--",
    "/bin/echo",
    "ok",
)

#: How long the probe may take before it is called a failure. A bwrap that
#: hangs is not a working bwrap.
PROBE_TIMEOUT_SECONDS: Final[float] = 10.0

#: The named fixes, in the order a user should try them. These strings are
#: rendered verbatim by the TUI badge and ``bevel doctor`` -- the loud part
#: of the loud degrade -- and are pinned by tests in both places.
_FIX_BWRAP: Final[str] = "sudo apt install bubblewrap  (provides /usr/bin/bwrap)"
_FIX_UIDMAP: Final[str] = "sudo apt install uidmap  (provides newuidmap/newgidmap)"
_FIX_APPARMOR: Final[str] = (
    "AppArmor: add a userns profile for bwrap at /etc/apparmor.d/bwrap "
    "(Ubuntu 24.04 restricts unprivileged user namespaces; "
    "see https://github.com/containers/bubblewrap)"
)


@dataclass(frozen=True)
class ProbeResult:
    """One probe run. ``passed`` is the only field callers branch on.

    ``spawn_error`` distinguishes "the probe could not execute" from "the
    probe executed and the kernel said no" -- the same distinction
    :class:`bevel._capability.Capability` draws with its ``error`` field.
    """

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    spawn_error: str | None

    @property
    def passed(self) -> bool:
        return (
            self.spawn_error is None
            and not self.timed_out
            and self.exit_code == 0
            and self.stdout.strip() == "ok"
        )

    @property
    def fixes(self) -> tuple[str, ...]:
        """The named fixes for *this* failure, chosen from what it actually said.

        ``setting up uid map: Permission denied`` is the signature both the
        missing-uidmap case and the missing-AppArmor-profile case print, so
        both are named -- they are cheap to check and each has been the
        culprit on a real machine.
        """
        if self.spawn_error is not None or self.timed_out:
            return (_FIX_BWRAP,)
        combined = f"{self.stderr}\n{self.stdout}".lower()
        if "uid map" in combined or "uidmap" in combined:
            return (_FIX_UIDMAP, _FIX_APPARMOR)
        return (_FIX_BWRAP, _FIX_UIDMAP, _FIX_APPARMOR)


def resolve_bwrap() -> Path | None:
    """The resolved path to bwrap, or None when it is not installed.

    Resolved, not the PATH name: the AppArmor grant is by path, so spawning
    a bare ``bwrap`` risks executing an uncovered copy that will fail in a
    way that looks like a broken install.
    """
    found = shutil.which("bwrap")
    if found is None:
        return None
    return Path(found).resolve(strict=False)


def build_prefix(bwrap_path: Path, workspace: Path) -> tuple[str, ...]:
    """The argv prefix the shell tool prepends to every spawned command.

    Order matters and is pinned by tests: ``--ro-bind / /`` first, then the
    writable bind of the workspace over its own path, then the pseudo-
    filesystems and the two unshares, then ``--`` so everything after is the
    tool's argv and never a bwrap option.
    """
    ws = str(workspace)
    return (
        str(bwrap_path),
        "--ro-bind",
        "/",
        "/",
        "--bind",
        ws,
        ws,
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unshare-net",
        "--unshare-user",
        "--die-with-parent",
        "--",
    )


def run_probe(bwrap_path: Path) -> ProbeResult:
    """Run the documented probe and report exactly what happened.

    Synchronous and side-effect free apart from the child it spawns; called
    at startup by the front end, not per tool call. ``subprocess.run`` with
    an argv list -- never ``shell=True``, the AGENTS.md rule that applies to
    every spawn in the codebase including this one.
    """
    argv = [str(bwrap_path), *PROBE_ARGV]
    try:
        completed = subprocess.run(  # noqa: S603 - argv is bevel-built, never model-built
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(
            command=tuple(argv),
            exit_code=None,
            stdout="",
            stderr=f"the probe did not finish within {PROBE_TIMEOUT_SECONDS:g}s",
            timed_out=True,
            spawn_error=None,
        )
    except OSError as exc:
        return ProbeResult(
            command=tuple(argv),
            exit_code=None,
            stdout="",
            stderr=str(exc),
            timed_out=False,
            spawn_error=f"{type(exc).__name__}: {exc}",
        )
    return ProbeResult(
        command=tuple(argv),
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
        spawn_error=None,
    )


def _probe_degrade(result: ProbeResult) -> DegradeReport:
    """The loud degrade for a probe that ran and failed (or could not run)."""
    reason = (
        "the bwrap probe could not be executed"
        if result.spawn_error is not None
        else "the bwrap probe did not complete"
    )
    return DegradeReport(
        reason=reason,
        command=" ".join(result.command),
        exit_code=result.exit_code,
        stderr=result.stderr.strip(),
        fixes=result.fixes,
    )


async def detect_sandbox(workspace: Path) -> SandboxStatus:
    """Decide the enforcement level for this process: probe, self-test, report.

    The ladder goes down, never up, and every step down records why:

    1. Not Linux -> POLICY_ONLY with ``probe_attempted=False``. m12a adds no
       Windows behavior; m12b owns that decision, and until it does the
       absence of a probe must not render as a measured degrade.
    2. bwrap absent -> POLICY_ONLY naming ``sudo apt install bubblewrap``.
    3. Probe fails -> POLICY_ONLY with the probe's command, exit code,
       stderr, and named fixes.
    4. Probe passes but the self-test's forbidden write or connect was *not*
       denied -> POLICY_ONLY naming the self-test: the probe lied.
    5. Probe passes and the self-test watched the kernel deny both -> BWRAP.

    Never raises: a sandbox layer that can crash the front end is a front end
    that starts without a sandbox and says nothing, which is the failure this
    whole module exists to prevent.
    """
    if _CURRENT_PLATFORM != "linux":
        return SandboxStatus(
            level=EnforcementLevel.POLICY_ONLY,
            prefix=(),
            probe_attempted=False,
            degrade=None,
        )

    path = resolve_bwrap()
    if path is None:
        return SandboxStatus(
            level=EnforcementLevel.POLICY_ONLY,
            prefix=(),
            probe_attempted=True,
            degrade=DegradeReport(
                reason="bwrap is not installed",
                command=None,
                exit_code=None,
                stderr="",
                fixes=(_FIX_BWRAP,),
            ),
        )

    result = run_probe(path)
    if not result.passed:
        return SandboxStatus(
            level=EnforcementLevel.POLICY_ONLY,
            prefix=(),
            probe_attempted=True,
            degrade=_probe_degrade(result),
        )

    prefix = build_prefix(path, workspace)
    report: SelfTestReport = await run_selftest(prefix, workspace)
    if not report.ok:
        return SandboxStatus(
            level=EnforcementLevel.POLICY_ONLY,
            prefix=(),
            probe_attempted=True,
            degrade=DegradeReport(
                reason=(
                    "the bwrap probe passed but the self-test was not denied by the "
                    "kernel, so the probe lied"
                ),
                command=" ".join(result.command),
                exit_code=result.exit_code,
                stderr=report.detail,
                fixes=(_FIX_APPARMOR, _FIX_UIDMAP),
            ),
        )

    return SandboxStatus(
        level=EnforcementLevel.BWRAP,
        prefix=prefix,
        probe_attempted=True,
        degrade=None,
    )

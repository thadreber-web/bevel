"""Read-only probes for the sandbox features this machine actually has.

This module never *enforces* anything and never mutates state. It answers one
question -- "if bevel asked for this isolation level, would the OS honour it?"
-- so that the sandbox layer (:mod:`bevel.core.sandbox`, milestone m12) can
pick a tier and, more importantly, *report the truth* about which tier it got.

Keeping the probes here rather than inside the backends means ``bevel doctor``
works before any enforcement code exists, and the probes stay side-effect free
and therefore trivially testable.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Final

#: Windows editions that ship Windows Sandbox. Home/SingleLanguage do not.
_SANDBOX_CAPABLE_EDITIONS: Final[frozenset[str]] = frozenset(
    {
        "Professional",
        "ProfessionalEducation",
        "ProfessionalWorkstation",
        "ProfessionalWorkstationN",
        "Enterprise",
        "EnterpriseS",
        "EnterpriseG",
        "EnterpriseGEval",
        "Education",
        "ServerRdsh",
        "CloudEdition",
        "CloudUser",
    }
)


#: ``sys.platform`` behind an opaque ``str``. mypy special-cases comparisons
#: against ``sys.platform`` and, with ``warn_unreachable`` on, marks the
#: other OS's branches dead -- which breaks whichever typecheck lane is not
#: the host's. The typecheck lanes must pass identically on ubuntu and
#: windows, so the probes are checked as written on both.
_CURRENT_PLATFORM: Final[str] = sys.platform


@dataclass(frozen=True)
class Capability:
    """One probe result.

    ``available`` is the only field callers should branch on. ``detail`` is a
    short human-readable string for ``bevel doctor`` -- it must never be
    parsed programmatically, and it must never overstate what was found.
    """

    name: str
    available: bool
    detail: str
    #: Set when a probe could not be completed, as opposed to completing and
    #: finding the feature absent. ``available`` is False in both cases; the
    #: distinction matters to ``doctor`` output and to bug reports.
    error: str | None = None


@dataclass(frozen=True)
class CapabilityReport:
    """Everything ``bevel doctor`` needs to know about this machine."""

    platform: str
    release: str
    machine: str
    python_version: str
    in_container: bool
    capabilities: tuple[Capability, ...] = field(default_factory=tuple)

    def get(self, name: str) -> Capability | None:
        for cap in self.capabilities:
            if cap.name == name:
                return cap
        return None

    @property
    def best_tier(self) -> int:
        """Highest sandbox tier this machine could plausibly reach.

        0 = policy-only, 1 = in-process kernel enforcement, 2 = OS container.
        This is an *upper bound from static probing* -- the runtime self-test
        in m12 is what actually decides, because only it can verify that a
        forbidden operation is genuinely denied.
        """
        tier = 0
        for name in ("landlock", "appcontainer"):
            cap = self.get(name)
            if cap is not None and cap.available:
                tier = max(tier, 1)
        for name in ("windows_sandbox", "bwrap"):
            cap = self.get(name)
            if cap is not None and cap.available:
                tier = max(tier, 2)
        return tier


def _probe_landlock() -> Capability:
    """Ask the kernel for its Landlock ABI version.

    ``landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)`` is the
    documented way to query support without creating anything. A return of -1
    means Landlock is off or too old.
    """
    if _CURRENT_PLATFORM != "linux":
        return Capability("landlock", False, "not Linux")
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        try:
            fn = libc.landlock_create_ruleset
        except AttributeError:
            return Capability("landlock", False, "libc has no landlock_create_ruleset")
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
        version = fn(None, 0, 1 << 0)  # LANDLOCK_CREATE_RULESET_VERSION
        if version < 0:
            errno = ctypes.get_errno()
            return Capability("landlock", False, f"kernel refused (errno {errno})")
        return Capability("landlock", True, f"ABI v{version}")
    except OSError as exc:
        return Capability("landlock", False, "could not load libc", error=str(exc))


def _probe_seccomp() -> Capability:
    """Check that the kernel exposes seccomp filters.

    ``/proc/sys/kernel/seccomp/actions_avail`` only exists when CONFIG_SECCOMP
    was built in, which is the whole question we care about at probe time.
    """
    if _CURRENT_PLATFORM != "linux":
        return Capability("seccomp", False, "not Linux")
    path = "/proc/sys/kernel/seccomp/actions_avail"
    try:
        with open(path, encoding="utf-8") as fh:
            actions = fh.read().split()
    except FileNotFoundError:
        return Capability("seccomp", False, "seccomp not built into kernel")
    except OSError as exc:
        return Capability("seccomp", False, "could not read seccomp status", error=str(exc))
    if not actions:
        return Capability("seccomp", False, "no seccomp actions available")
    return Capability("seccomp", True, f"actions: {', '.join(actions[:4])}")


def _windows_edition() -> str | None:
    try:
        import winreg  # noqa: PLC0415 -- Windows-only import
    except ImportError:
        return None

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "EditionID")
    except OSError:
        return None
    return str(value) if value else None


def _windows_sandbox_service_installed() -> bool:
    """True when the Windows Sandbox service is registered with the SCM.

    Distinguishes a machine where the feature is enabled from one where the
    binaries are merely staged and ``WindowsSandbox.exe`` would fail to start.
    """
    try:
        import winreg  # noqa: PLC0415 -- Windows-only import
    except ImportError:
        return False

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services\WindowsSandboxService"
        ):
            return True
    except OSError:
        return False


def _probe_appcontainer() -> Capability:
    """AppContainer is Chromium/Edge's sandbox primitive, and bevel's tier-1 backend.

    Two exports are required and both are probed, because they do different
    jobs and live in different DLLs:

    ``userenv!CreateAppContainerProfile``
        creates the named profile and derives the AppContainer SID.
    ``kernelbase!CreateAppContainerToken``
        caps a token with that SID; this is what actually launches the tool.

    Export locations were confirmed by parsing PE export tables rather than
    trusting documentation: the profile APIs are exported by ``userenv.dll``
    and ``CreateAppContainerToken`` by ``KernelBase.dll`` -- kernel32 does
    *not* forward it, so probing kernel32 gives a false negative. They are
    not in ole32 either, and the ``api-ms-win-core-appcontainer-*`` API-set
    DLLs do not exist as files on this build, so loading those names by hand
    fails with error 126.

    Available on every modern Windows SKU including Home, unlike Windows
    Sandbox. Static probing only -- actual enforcement is verified by m12.
    """
    if _CURRENT_PLATFORM != "win32":
        return Capability("appcontainer", False, "not Windows")
    required = (("userenv", "CreateAppContainerProfile"), ("kernelbase", "CreateAppContainerToken"))
    missing: list[str] = []
    for dll, symbol in required:
        try:
            handle = ctypes.CDLL(dll)
        except OSError as exc:
            missing.append(f"{dll}!{symbol} ({exc})")
            continue
        if getattr(handle, symbol, None) is None:
            missing.append(f"{dll}!{symbol}")
    if missing:
        return Capability("appcontainer", False, "missing " + ", ".join(missing))
    return Capability("appcontainer", True, "userenv+kernelbase appcontainer exports present")


def _probe_job_object() -> Capability:
    if _CURRENT_PLATFORM != "win32":
        return Capability("job_object", False, "not Windows")
    try:
        kernel32 = ctypes.CDLL("kernel32")
    except OSError as exc:
        return Capability("job_object", False, "could not load kernel32", error=str(exc))
    if not hasattr(kernel32, "CreateJobObjectW"):
        return Capability("job_object", False, "CreateJobObjectW missing")
    return Capability("job_object", True, "CreateJobObjectW present")


def _probe_windows_sandbox() -> Capability:
    """Windows Sandbox: hypervisor-backed, but disposable and Pro-or-better only.

    bevel treats this as an opt-in tier for untrusted repositories, not the
    default execution path -- a 10-30s cold start per tool call is unusable,
    and the design is throwaway while an agent must write real edits.

    ``WindowsSandbox.exe`` ships staged on capable SKUs whether or not the
    ``Containers-DisposableClientVM`` feature is enabled, so its existence
    alone is not enough to claim the tier. The service is what proves the
    feature is actually turned on; without it the exe fails to launch.
    """
    if _CURRENT_PLATFORM != "win32":
        return Capability("windows_sandbox", False, "not Windows")
    # `SystemRoot` is the documented variable name; Windows env lookups are
    # case-insensitive, so SIM112's capitalised suggestion is not an improvement.
    windir = os.environ.get("SystemRoot", r"C:\Windows")  # noqa: SIM112
    system32 = os.path.join(windir, "System32")
    exe = os.path.join(system32, "WindowsSandbox.exe")
    if not os.path.exists(exe):
        return Capability("windows_sandbox", False, "WindowsSandbox.exe not present")
    edition = _windows_edition()
    if edition and edition not in _SANDBOX_CAPABLE_EDITIONS:
        return Capability("windows_sandbox", False, f"edition {edition} lacks Sandbox")
    if not _windows_sandbox_service_installed():
        return Capability(
            "windows_sandbox",
            False,
            "binaries staged but WindowsSandboxService is not installed; "
            "enable the Containers-DisposableClientVM feature",
        )
    if edition is None:
        return Capability("windows_sandbox", True, "service installed, edition unknown")
    return Capability("windows_sandbox", True, f"service installed on {edition}")


def _probe_bwrap() -> Capability:
    """bubblewrap is an optional enhancement, never a requirement.

    The user's Spark must not need any package installed for bevel to reach
    tier 1, so bwrap is detected and used if present -- never installed.
    """
    if _CURRENT_PLATFORM != "linux":
        return Capability("bwrap", False, "not Linux")
    path = shutil.which("bwrap")
    if path is None:
        return Capability("bwrap", False, "bwrap not on PATH (optional)")
    return Capability("bwrap", True, path)


def _in_container() -> bool:
    """Best-effort container detection; affects what a sandbox can promise."""
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as fh:
            body = fh.read()
    except OSError:
        return False
    return any(marker in body for marker in ("docker", "kubepods", "containerd", "/lxc/"))


def _release() -> str:
    """A release string that is actually identifying.

    ``platform.release()`` returns ``"10"`` on Windows 11 -- the whole OS
    reports 10.0 for compatibility. ``platform.version()`` carries the build
    (``10.0.26200``), which is what distinguishes sandbox behaviour between
    Windows releases, so it is used there. On Linux the reverse holds:
    ``release()`` is the useful kernel version and ``version()`` is a build
    stamp.
    """
    if _CURRENT_PLATFORM == "win32":
        return platform.version() or platform.release()
    return platform.release()


def probe() -> CapabilityReport:
    """Run every probe and return a report. Never raises."""
    probes = (
        _probe_landlock,
        _probe_seccomp,
        _probe_appcontainer,
        _probe_job_object,
        _probe_windows_sandbox,
        _probe_bwrap,
    )
    return CapabilityReport(
        platform=platform.system(),
        release=_release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        in_container=_in_container(),
        capabilities=tuple(probe_fn() for probe_fn in probes),
    )

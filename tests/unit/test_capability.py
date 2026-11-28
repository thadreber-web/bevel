"""Tests for the read-only capability probes.

These assert the honesty contract: a probe that could not confirm a feature
must report ``available=False``, and ``best_tier`` must never rise above what
the probes actually confirmed. Platform-specific branches are exercised by
monkeypatching, so the same assertions hold on the ubuntu and windows CI
lanes.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import pytest

from bevel import _capability
from bevel._capability import Capability, CapabilityReport


def _report(*caps: Capability) -> CapabilityReport:
    return CapabilityReport(
        platform="TestOS",
        release="0",
        machine="test",
        python_version="3.11.0",
        in_container=False,
        capabilities=caps,
    )


def _cap(name: str, available: bool) -> Capability:
    return Capability(name, available, "test")


class TestBestTier:
    def test_nothing_available_is_policy_only(self) -> None:
        assert _report().best_tier == 0

    def test_appcontainer_reaches_tier_one(self) -> None:
        assert _report(_cap("appcontainer", True)).best_tier == 1

    def test_landlock_reaches_tier_one(self) -> None:
        assert _report(_cap("landlock", True)).best_tier == 1

    def test_os_container_reaches_tier_two(self) -> None:
        assert _report(_cap("windows_sandbox", True)).best_tier == 2

    def test_unconfirmed_capability_does_not_raise_tier(self) -> None:
        """The point of the whole module: absence must not look like protection."""
        report = _report(
            _cap("appcontainer", False),
            _cap("windows_sandbox", False),
            _cap("landlock", False),
            _cap("bwrap", False),
        )
        assert report.best_tier == 0

    def test_tier_is_the_maximum_not_the_last(self) -> None:
        report = _report(_cap("windows_sandbox", True), _cap("appcontainer", True))
        assert report.best_tier == 2


class TestGet:
    def test_missing_capability_is_none(self) -> None:
        assert _report(_cap("landlock", True)).get("appcontainer") is None


class FakeDll:
    """A ctypes DLL stand-in whose visible exports are chosen by the test."""

    def __init__(self, names: set[str]) -> None:
        self._names = names

    def __getattr__(self, item: str) -> Any:
        if item in self._names:
            return object()
        raise AttributeError(item)


class TestAppContainerProbe:
    def test_not_windows_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "linux")
        cap = _capability._probe_appcontainer()
        assert cap.available is False
        assert cap.detail == "not Windows"

    def test_missing_token_export_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """kernel32 does not forward CreateAppContainerToken; probing it must fail loudly.

        This is the exact bug this probe used to have -- it looked in ole32 for
        CreateAppContainerProfile and reported a capable machine as incapable.
        """

        def fake_cdll(name: str) -> FakeDll:
            exports = {"CreateAppContainerProfile"} if name == "userenv" else set()
            return FakeDll(exports)

        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

        cap = _capability._probe_appcontainer()
        assert cap.available is False
        assert "kernelbase!CreateAppContainerToken" in cap.detail

    def test_both_exports_present_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        exports = {
            "userenv": {"CreateAppContainerProfile"},
            "kernelbase": {"CreateAppContainerToken"},
        }

        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeDll(exports[name]))

        assert _capability._probe_appcontainer().available is True

    def test_unloadable_dll_is_reported_as_error_not_silence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(name: str) -> Any:
            raise OSError(f"cannot load {name}")

        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(ctypes, "CDLL", boom)

        cap = _capability._probe_appcontainer()
        assert cap.available is False
        assert "userenv!CreateAppContainerProfile" in cap.detail


class TestWindowsSandboxProbe:
    """Staged binaries are not an enabled feature -- claiming tier 2 on them lies."""

    def _stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        exe_exists: bool,
        service_installed: bool,
        edition: str | None,
    ) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(_capability.os.path, "exists", lambda _p: exe_exists)
        monkeypatch.setattr(
            _capability, "_windows_sandbox_service_installed", lambda: service_installed
        )
        monkeypatch.setattr(_capability, "_windows_edition", lambda: edition)

    def test_absent_binaries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stage(monkeypatch, exe_exists=False, service_installed=False, edition="Professional")
        assert _capability._probe_windows_sandbox().available is False

    def test_staged_but_service_missing_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stage(monkeypatch, exe_exists=True, service_installed=False, edition="Professional")
        cap = _capability._probe_windows_sandbox()
        assert cap.available is False
        assert "Containers-DisposableClientVM" in cap.detail

    def test_service_installed_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stage(monkeypatch, exe_exists=True, service_installed=True, edition="Professional")
        assert _capability._probe_windows_sandbox().available is True

    def test_home_edition_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stage(monkeypatch, exe_exists=True, service_installed=True, edition="Home")
        cap = _capability._probe_windows_sandbox()
        assert cap.available is False
        assert "Home" in cap.detail


class TestWinregAccess:
    """winreg is imported dynamically so the Linux typecheck lane can resolve it.

    typeshed guards winreg.pyi behind sys.platform == "win32", so a static
    `import winreg` is an attr-defined error on ubuntu and a `# type: ignore`
    for it would be an unused-ignore error on windows. Dynamic import means
    neither lane lies, but the degradation path now needs its own tests.
    """

    def test_missing_winreg_degrades_rather_than_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeImportlib:
            @staticmethod
            def import_module(name: str) -> Any:
                raise ImportError(name)

        monkeypatch.setattr(_capability, "importlib", FakeImportlib())
        assert _capability._winreg() is None
        assert _capability._windows_edition() is None
        assert _capability._windows_sandbox_service_installed() is False

    def test_registry_failure_is_not_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A machine where the key read fails must not be reported as capable."""

        class ExplodingWinreg:
            HKEY_LOCAL_MACHINE = "HKLM"

            @staticmethod
            def OpenKey(*_args: Any) -> Any:
                raise OSError(2, "file not found")

        monkeypatch.setattr(_capability, "_winreg", ExplodingWinreg)
        assert _capability._windows_sandbox_service_installed() is False
        assert _capability._windows_edition() is None


class TestRelease:
    def test_windows_reports_build_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """platform.release() says "10" on Windows 11, which identifies nothing."""
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(_capability.platform, "version", lambda: "10.0.26200")
        monkeypatch.setattr(_capability.platform, "release", lambda: "10")
        assert _capability._release() == "10.0.26200"

    def test_windows_falls_back_to_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        monkeypatch.setattr(_capability.platform, "version", lambda: "")
        monkeypatch.setattr(_capability.platform, "release", lambda: "10")
        assert _capability._release() == "10"

    def test_linux_reports_kernel_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "linux")
        monkeypatch.setattr(_capability.platform, "version", lambda: "#1-Ubuntu")
        monkeypatch.setattr(_capability.platform, "release", lambda: "6.8.0")
        assert _capability._release() == "6.8.0"


class TestProbe:
    def test_probe_never_raises_and_covers_every_capability(self) -> None:
        report = _capability.probe()
        names = {cap.name for cap in report.capabilities}
        assert names == {
            "landlock",
            "seccomp",
            "appcontainer",
            "job_object",
            "windows_sandbox",
            "bwrap",
        }
        assert report.platform
        assert report.python_version

    def test_off_platform_probes_are_unavailable(self) -> None:
        """Whichever OS this is, the other OS's features must read as absent."""
        report = _capability.probe()
        if _capability._CURRENT_PLATFORM == "linux":
            assert report.get("appcontainer") is not None
            assert report.get("appcontainer").available is False
        else:
            assert report.get("landlock") is not None
            assert report.get("landlock").available is False
            assert report.get("seccomp") is not None
            assert report.get("seccomp").available is False


class TestBwrapProbe:
    """The probe RUNS bwrap. shutil.which alone is the forbidden shortcut.

    On Ubuntu 24.04 *installed is not working*: the uidmap helpers and an
    AppArmor userns profile are separate prerequisites, and each has failed
    independently on a machine where ``bwrap`` was "installed". A probe that
    only searches PATH reports those machines as capable, which is the lie
    this probe exists to prevent.
    """

    def _linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "linux")

    def test_off_linux_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_capability, "_CURRENT_PLATFORM", "win32")
        cap = _capability._probe_bwrap()
        assert cap.available is False
        assert cap.detail == "not Linux"

    def test_absent_bwrap_is_unavailable_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent and error are distinct cases; both unavailable, only one is an error."""
        self._linux(monkeypatch)
        monkeypatch.setattr(_capability.shutil, "which", lambda _name: None)
        cap = _capability._probe_bwrap()
        assert cap.available is False
        assert cap.error is None
        assert "not on PATH" in cap.detail

    def test_installed_but_broken_is_unavailable_with_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Spark's actual failure: bwrap on PATH, uid map refused."""
        self._linux(monkeypatch)
        monkeypatch.setattr(_capability.shutil, "which", lambda _name: "/usr/bin/bwrap")
        monkeypatch.setattr(
            _capability.bwrap,
            "run_probe",
            lambda _path: _capability.bwrap.ProbeResult(
                command=("/usr/bin/bwrap", *_capability.bwrap.PROBE_ARGV),
                exit_code=1,
                stdout="",
                stderr="bwrap: setting up uid map: Permission denied",
                timed_out=False,
                spawn_error=None,
            ),
        )
        cap = _capability._probe_bwrap()
        assert cap.available is False
        assert "uid map" in cap.detail
        assert cap.error is None  # the probe completed; it completed and said no

    def test_probe_that_could_not_run_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe that could not execute is a different fact than one that said no."""
        self._linux(monkeypatch)
        monkeypatch.setattr(_capability.shutil, "which", lambda _name: "/usr/bin/bwrap")
        monkeypatch.setattr(
            _capability.bwrap,
            "run_probe",
            lambda _path: _capability.bwrap.ProbeResult(
                command=("/usr/bin/bwrap", *_capability.bwrap.PROBE_ARGV),
                exit_code=None,
                stdout="",
                stderr="",
                timed_out=False,
                spawn_error="PermissionError: not executable",
            ),
        )
        cap = _capability._probe_bwrap()
        assert cap.available is False
        assert cap.error is not None
        assert "PermissionError" in cap.error

    def test_working_bwrap_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._linux(monkeypatch)
        monkeypatch.setattr(_capability.shutil, "which", lambda _name: "/usr/bin/bwrap")
        monkeypatch.setattr(
            _capability.bwrap,
            "run_probe",
            lambda _path: _capability.bwrap.ProbeResult(
                command=("/usr/bin/bwrap", *_capability.bwrap.PROBE_ARGV),
                exit_code=0,
                stdout="ok\n",
                stderr="",
                timed_out=False,
                spawn_error=None,
            ),
        )
        cap = _capability._probe_bwrap()
        assert cap.available is True
        assert cap.error is None


class TestInContainer:
    def test_absent_cgroup_is_not_a_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe that cannot read /proc must not claim containment, and must not raise."""
        real_open = open

        def fake_open(path: object, **kwargs: Any) -> Any:
            if path == "/proc/1/cgroup":
                raise FileNotFoundError(path)
            return real_open(path, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        assert _capability._in_container() is False

    def test_docker_cgroup_is_detected(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cgroup = tmp_path / "cgroup"
        cgroup.write_text("0::/system.slice/docker-abc.scope\n", encoding="utf-8")
        real_open = open

        def fake_open(path: object, **kwargs: Any) -> Any:
            if path == "/proc/1/cgroup":
                return real_open(cgroup, **kwargs)
            raise AssertionError(f"unexpected read of {path}")

        monkeypatch.setattr("builtins.open", fake_open)
        assert _capability._in_container() is True

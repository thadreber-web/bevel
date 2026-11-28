"""The two enforcement levels that exist, and the record of which one was verified.

m12a deleted the old tier ladder. There is no Landlock backend, no seccomp
backend, no AppContainer backend, no Windows Sandbox backend, and no enum
slot reserved for one -- an enum variant for a backend that will never ship
is an aspirational ``--help`` text in type form, which AGENTS.md names as the
worst failure mode. What ships is exactly two states:

``POLICY_ONLY``
    The T0 path jail (:mod:`bevel.core.tools.base`) and the rule engine
    (:mod:`bevel.core.policy`). Always on, both platforms, the only thing
    that protects you when bwrap is absent or broken.

``BWRAP``
    The shell tool spawns through a ``bwrap`` argv prefix and the kernel
    enforces read-only-except-workspace with no network. This level is
    *earned*, never requested: it is set only after the probe ran bwrap for
    real and the self-test watched a real child get denied by the kernel
    (:mod:`bevel.core.sandbox.selftest`).

The honesty rule this module encodes: the UI reports the level that was
**verified**, never the level that was requested or configured. A machine
with bwrap installed but broken (the Ubuntu 24.04 ``uidmap``/AppArmor case)
is ``POLICY_ONLY`` with a :class:`DegradeReport` that says exactly why --
and every consumer of that report (the TUI badge, ``bevel doctor``) must
render it loudly. A ``POLICY_ONLY`` with no reason attached is a bug in
whoever produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["DegradeReport", "EnforcementLevel", "SandboxStatus"]


class EnforcementLevel(StrEnum):
    """The enforcement bevel *verified*, not the enforcement bevel was asked for.

    Exactly two states. A third variant would mean a backend exists; if you
    are adding one, the plan changed and this docstring is the place that
    says so.
    """

    POLICY_ONLY = "policy-only"
    BWRAP = "bwrap"


@dataclass(frozen=True)
class DegradeReport:
    """Why the level is lower than bwrap, in the words a user can act on.

    The loud-degrade contract (the non-negotiable part of m12a): ``command``
    is the probe command *verbatim*, ``exit_code`` and ``stderr`` are what it
    actually returned, and ``fixes`` names the missing prerequisite and the
    exact install -- the same register as ``gh auth login`` in m11b and the
    missing ``user.name`` in the git tool. An opaque degrade is the silent
    fake this project forbids, so every field is rendered by every consumer.
    """

    #: One-sentence account of what happened, e.g. "the probe did not complete".
    reason: str
    #: The probe command exactly as bevel ran it, or None when bevel never
    #: got that far (bwrap absent) -- in which case the fixes still name the
    #: install and the probe command is the documented one.
    command: str | None
    #: The probe's real exit code; None when the child never ran or never exited.
    exit_code: int | None
    #: The probe's real stderr. Empty only when there genuinely was none.
    stderr: str
    #: The named fixes, in the order a user should try them.
    fixes: tuple[str, ...]


@dataclass(frozen=True)
class SandboxStatus:
    """What :func:`bevel.core.sandbox.bwrap.detect_sandbox` decided, and why.

    Produced once per process by the front end (the TUI, ``bevel doctor``)
    and handed down: the shell tool reads :attr:`prefix`, the TUI renders
    :attr:`badge`, and ``doctor`` renders the whole thing. Nothing else gets
    to decide the level.
    """

    level: EnforcementLevel
    #: The bwrap argv prefix to spawn through, empty when there is none.
    prefix: tuple[str, ...]
    #: False only on platforms where m12a never probed (Windows, until m12b
    #: makes the decision there). The badge renders nothing when False:
    #: "sandbox: policy-only" would claim a degrade that was never measured.
    probe_attempted: bool
    #: Set exactly when the level is POLICY_ONLY *because of bwrap* -- probe
    #: absent, probe failed, or probe lied and the self-test caught it.
    degrade: DegradeReport | None

    @property
    def badge(self) -> str | None:
        """The TUI badge text, or None when no probe ran to justify a badge."""
        if not self.probe_attempted:
            return None
        return f"sandbox: {self.level.value}"

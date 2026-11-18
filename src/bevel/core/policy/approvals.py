"""The approval seam: what the engine asks, and who is allowed to answer.

When a call is neither free nor outright denied, the engine has a question:
*may this run?* This module owns the shape of that question and the two answers
that exist today. The decision itself belongs to whoever implements
:class:`Approver` — the Textual modal in m07 asks a human, and the human's
answer (including "always allow", which the engine then learns) is what
decides.

The two approvers here are what a run does when no human is in the loop:
:class:`DenyAllApprover` refuses and says plainly that nobody was there to ask,
and :class:`AutoApprover` allows everything for callers that decided in advance
that nothing needs asking. Neither pretends to be the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from bevel.core.tools.base import RiskTier

__all__ = [
    "Approval",
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "Decision",
    "DenyAllApprover",
]


class Decision(StrEnum):
    """What policy finally said about one tool call."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class ApprovalRequest:
    """One question from the engine to an approver.

    ``argv`` is set for command calls and is what a learned rule is keyed on;
    ``paths`` is what an out-of-workspace write needs explained. ``reason`` is
    the tier's own justification, so an approver can show a human *why* it is
    being interrupted instead of just "allow? (y/n)".
    """

    name: str
    tier: RiskTier
    reason: str
    argv: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()

    @property
    def needs_typed_confirmation(self) -> bool:
        """Destructive calls are not satisfied by a yes/no keypress."""
        return self.tier is RiskTier.EXEC_DESTRUCTIVE

    @property
    def confirmation(self) -> str:
        """The exact text a human must type to approve a destructive call.

        Typing the command back is the point: it is the difference between
        acknowledging a prompt and reading the prompt.
        """
        return " ".join(self.argv) if self.argv else self.name


@dataclass(frozen=True)
class Approval:
    """An approver's answer.

    ``typed`` is what the human actually entered for a destructive call; the
    engine, not the approver, decides whether it matched. ``remember`` asks the
    engine to persist the call's argv as a learned allowlist entry, which is
    how "always allow ``pytest``" reaches ``policy.db``.
    """

    decision: Decision
    note: str = ""
    typed: str = ""
    remember: bool = False


class Approver(Protocol):
    """Something that can answer :class:`ApprovalRequest`.

    Two implementations exist and the engine depends only on this: the m07 TUI
    modal will be a third. It is a protocol rather than a base class because
    the engine must never learn what an approver is built on.
    """

    def approve(self, request: ApprovalRequest) -> Approval: ...


class DenyAllApprover:
    """The headless default: there is nobody at the keyboard, so nothing is approved.

    The note is written for the model to read — it says what was asked, who was
    missing, and what would let the call through, so the next iteration can
    adapt instead of retrying a call that will be refused forever.
    """

    def approve(self, request: ApprovalRequest) -> Approval:
        asked = " ".join(request.argv) if request.argv else request.name
        return Approval(
            Decision.DENY,
            note=(
                f"{request.name!r} ({request.tier.value}) needs approval before it runs, "
                f"and no approver is installed to give it: nothing was asked of you. "
                f"Ask the user to approve {asked!r}, or run with an approver that answers "
                "automatically."
            ),
        )


class AutoApprover:
    """Approves every question it is asked, and says so in the log.

    For runs whose operator has already decided that approvals are not wanted.
    It satisfies typed confirmation on purpose — that is the whole point of
    opting out — but the decision log still records that an automatic approver
    answered, so a later read of the session can tell an approved call from an
    unexamined one.
    """

    def approve(self, request: ApprovalRequest) -> Approval:
        return Approval(
            Decision.ALLOW,
            note="approved automatically by AutoApprover; no human was consulted",
            typed=request.confirmation,
        )

"""Tiered policy engine and the approval protocol.

Seven risk tiers from READ (free) to EXEC_DESTRUCTIVE (always ask, typed
confirmation). Every decision is logged together with the rule that produced
it, so the TUI can explain itself.

The engine answers the agent loop with an allow or a deny; anything that has to
be *asked* goes to an :data:`Approver`. The m07 TUI modal plugs in there. With no
approver installed the engine is fail-closed — it uses :data:`DenyAllApprover`, so
an unanswered request is refused rather than granted. :data:`AutoApprover` is the
caller's explicit opt-out, and it records that no human was consulted.

This package never imports :mod:`bevel.core.agent` (the loop imports it) and
never imports :mod:`bevel.core.sandbox` (sandboxing is orthogonal to permission).
"""

from __future__ import annotations

from bevel.core.policy.approvals import (
    Approval,
    ApprovalRequest,
    Approver,
    AutoApprover,
    Decision,
    DenyAllApprover,
)
from bevel.core.policy.engine import (
    DecisionLog,
    PolicyDecision,
    PolicyEngine,
    PolicyEvaluator,
    PolicyRequest,
    PolicyVerdict,
    allow_all_policy,
)
from bevel.core.policy.rules import (
    LearnedRule,
    RuleStore,
    match_destructive,
    match_safe,
    normalise_argv,
    rule_hash,
)

__all__ = [
    "Approval",
    "ApprovalRequest",
    "Approver",
    "AutoApprover",
    "Decision",
    "DecisionLog",
    "DenyAllApprover",
    "LearnedRule",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyEvaluator",
    "PolicyRequest",
    "PolicyVerdict",
    "RuleStore",
    "allow_all_policy",
    "match_destructive",
    "match_safe",
    "normalise_argv",
    "rule_hash",
]

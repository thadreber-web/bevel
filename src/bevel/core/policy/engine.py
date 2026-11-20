"""The policy engine: tier in, verdict out, rule named.

One call, one decision, one reason. The engine never executes anything and never
refuses silently — every path through it ends in a :class:`PolicyVerdict` whose
``reason`` says which rule fired, and in a row in the decision log. A verdict
with no rule behind it is the failure mode this module exists to prevent.

The order of evaluation *is* the security property, so it is written down:

1. a config ``deny`` pattern refuses the call, whatever else is true;
2. for command calls, a built-in destructive pattern beats everything, including
   a learned allowlist entry — "always allow" can make a command free, it can
   never make ``rm -rf`` free;
3. a built-in safe pattern, a config ``allow`` pattern, or a learned rule makes a
   command ``EXEC_SAFE`` and it runs unasked;
4. otherwise the declared tier decides, with the path jail able to promote a
   workspace write to ``WRITE_OUTSIDE`` and the rule tables able to promote a
   command to ``EXEC_DESTRUCTIVE``.

Anything the engine cannot classify is refused, not allowed.
"""

from __future__ import annotations

import logging
import shlex
import sqlite3
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bevel.config.loader import policy_db_path
from bevel.config.schema import PolicyConfig, PolicyMode
from bevel.core.messages import ToolCall
from bevel.core.policy.approvals import (
    ApprovalRequest,
    Approver,
    Decision,
    DenyAllApprover,
)
from bevel.core.policy.rules import RuleStore, match_destructive, match_safe, normalise_argv
from bevel.core.tools.base import RiskTier, Tool, ToolContext, ToolError

logger = logging.getLogger(__name__)

__all__ = [
    "Decision",
    "PolicyDecision",
    "PolicyEngine",
    "PolicyEvaluator",
    "PolicyRequest",
    "PolicyVerdict",
    "allow_all_policy",
]

#: Tiers whose arguments carry a command to match against the rule tables.
_EXEC_TIERS = frozenset({RiskTier.EXEC_SAFE, RiskTier.EXEC_UNSAFE, RiskTier.EXEC_DESTRUCTIVE})

#: Tiers whose arguments carry a path to test against the workspace jail.
_PATH_TIERS = frozenset({RiskTier.WRITE_WORKSPACE, RiskTier.WRITE_OUTSIDE})


@dataclass(frozen=True)
class PolicyRequest:
    """What the loop asks the engine about: one repaired call, before execution."""

    call: ToolCall
    tool: Tool | None
    name: str
    arguments: Mapping[str, Any] | None


@dataclass(frozen=True)
class PolicyVerdict:
    """The engine's answer. ``reason`` is shown to the model when it denies."""

    decision: Decision
    reason: str = ""


PolicyEvaluator = Callable[[PolicyRequest], Awaitable[PolicyVerdict]]


async def allow_all_policy(request: PolicyRequest) -> PolicyVerdict:  # noqa: ARG001
    """Allow everything, unconditionally.

    Not the default — the default is :class:`PolicyEngine`. This is the explicit
    opt-out for a caller that has its own isolation (a container, a throwaway
    VM) and wants the loop to stop asking.
    """
    return PolicyVerdict(Decision.ALLOW, "policy disabled by the caller")


@dataclass(frozen=True)
class PolicyDecision:
    """One row of the decision log.

    ``rule`` is what makes the log worth reading: not "the write was allowed"
    but "the write was allowed because the mode was ``auto``", or "because the
    learned rule ``git status`` covered it".
    """

    at: str
    name: str
    tier: RiskTier
    decision: Decision
    rule: str
    reason: str = ""
    argv: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()


@dataclass
class _Draft:
    """The facts of a call, gathered once, before any rule looks at them."""

    name: str
    tier: RiskTier
    argv: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    outside: tuple[str, ...] = ()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    at       TEXT    NOT NULL,
    name     TEXT    NOT NULL,
    tier     TEXT    NOT NULL,
    decision TEXT    NOT NULL,
    rule     TEXT    NOT NULL,
    reason   TEXT    NOT NULL DEFAULT '',
    argv     TEXT    NOT NULL DEFAULT '',
    paths    TEXT    NOT NULL DEFAULT ''
)
"""


class DecisionLog:
    """Every decision the engine has made, in ``policy.db``.

    Persisted rather than kept in memory because the question the log answers —
    "why did that run touch that file?" — is usually asked after the run is
    over. The connection opens on first use, so an engine that never decides
    anything leaves no file behind.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._db_path)
            self._connection.execute(_SCHEMA)
            self._connection.commit()
        return self._connection

    def record(self, decision: PolicyDecision) -> None:
        connection = self._connect()
        connection.execute(
            "INSERT INTO decisions (at, name, tier, decision, rule, reason, argv, paths)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision.at,
                decision.name,
                decision.tier.value,
                decision.decision.value,
                decision.rule,
                decision.reason,
                shlex.join(decision.argv),
                shlex.join(decision.paths),
            ),
        )
        connection.commit()

    def recent(self, limit: int) -> tuple[PolicyDecision, ...]:
        """The most recent decisions, newest first."""
        rows = (
            self._connect()
            .execute(
                "SELECT at, name, tier, decision, rule, reason, argv, paths FROM decisions"
                " ORDER BY seq DESC LIMIT ?",
                (limit,),
            )
            .fetchall()
        )
        return tuple(
            PolicyDecision(
                at=at,
                name=name,
                tier=RiskTier(tier),
                decision=Decision(decision),
                rule=rule,
                reason=reason,
                argv=tuple(shlex.split(argv)) if argv else (),
                paths=tuple(shlex.split(paths)) if paths else (),
            )
            for at, name, tier, decision, rule, reason, argv, paths in rows
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class PolicyEngine:
    """The default :data:`PolicyEvaluator`.

    Callable, so it drops into the loop's ``policy`` seam unchanged. It holds the
    workspace jail (to classify writes), the rule store (for learned allowlists)
    and the approver (for everything that has to ask).

    SQLite is driven synchronously from inside the loop's event loop. That is a
    deliberate trade: one small write per tool call, into a file nothing else
    opens, is cheaper than the async wrapper would be.
    """

    def __init__(
        self,
        ctx: ToolContext,
        *,
        config: PolicyConfig | None = None,
        approver: Approver | None = None,
    ) -> None:
        self.ctx = ctx
        self.config = config if config is not None else PolicyConfig()
        self.approver: Approver = approver if approver is not None else DenyAllApprover()
        db_path = self.config.db_path if self.config.db_path is not None else policy_db_path()
        self.rules = RuleStore(db_path)
        self.log = DecisionLog(db_path)

    async def __call__(self, request: PolicyRequest) -> PolicyVerdict:
        verdict, record = await self._evaluate(request)
        self.log.record(record)
        if verdict.decision is Decision.DENY:
            logger.info(
                "policy denied %s (%s) by %s: %s",
                request.name,
                record.tier.value,
                record.rule,
                verdict.reason,
            )
        return verdict

    def decisions(self, limit: int = 50) -> tuple[PolicyDecision, ...]:
        """Recent decisions, newest first — what the TUI and ``bevel policy`` read."""
        return self.log.recent(limit)

    def remember(self, argv: Sequence[str]) -> None:
        """Persist ``always allow <cmd>`` without going through an approval."""
        self.rules.remember(tuple(str(item) for item in argv))

    def revoke(self, argv: Sequence[str]) -> bool:
        """Drop a learned rule. ``True`` if one was there to drop."""
        return self.rules.revoke(tuple(str(item) for item in argv))

    def close(self) -> None:
        self.rules.close()
        self.log.close()

    # ------------------------------------------------------------------ private

    async def _evaluate(self, request: PolicyRequest) -> tuple[PolicyVerdict, PolicyDecision]:
        tool = request.tool
        if tool is None:
            return self._deny(
                _Draft(request.name, RiskTier.EXEC_UNSAFE),
                rule="unknown-tool",
                reason=(
                    f"{request.name!r} is not a registered tool, so it has no declared risk "
                    "tier. The engine refuses calls it cannot classify."
                ),
            )

        draft = self._draft(request.name, tool.risk_tier, request.arguments or {})

        pattern = _match_patterns(draft.argv, self.config.deny)
        if pattern is not None:
            return self._deny(
                draft,
                rule=f"config:deny:{pattern}",
                reason=(
                    f"{shlex.join(draft.argv)} matches the deny rule {pattern!r} from your "
                    "configuration. Remove it from [policy] deny if that is wrong."
                ),
            )

        if draft.tier in _EXEC_TIERS:
            return await self._evaluate_exec(draft)
        if draft.tier is RiskTier.READ:
            return self._allow(draft, rule="tier:read", reason="reads are free")
        if draft.tier in _PATH_TIERS:
            return await self._evaluate_write(draft)
        return await self._ask(
            draft,
            rule=f"tier:{draft.tier.value}",
            reason=f"{draft.name} is tier {draft.tier.value}, which always needs your approval",
        )

    def _draft(self, name: str, tier: RiskTier, arguments: Mapping[str, Any]) -> _Draft:
        argv = _argv_of(arguments) if tier in _EXEC_TIERS else ()
        paths = _paths_of(arguments) if tier in _PATH_TIERS else ()
        outside = self._outside(paths) if tier in _PATH_TIERS else ()
        return _Draft(name=name, tier=tier, argv=argv, paths=paths, outside=outside)

    async def _evaluate_exec(self, draft: _Draft) -> tuple[PolicyVerdict, PolicyDecision]:
        if not draft.argv:
            return self._deny(
                draft,
                rule="exec:unclassifiable",
                reason=(
                    f"{draft.name} is a command-tier tool but did not supply a command, so "
                    "there is nothing to match against the rules. Refused rather than trusted."
                ),
            )

        destructive = match_destructive(draft.argv)
        if destructive is not None:
            draft = _Draft(draft.name, RiskTier.EXEC_DESTRUCTIVE, draft.argv, draft.paths)
            return await self._ask(
                draft,
                rule=f"builtin:{destructive}",
                reason=f"{shlex.join(draft.argv)} matches the destructive pattern {destructive!r}",
            )

        covered = match_safe(draft.argv) or _match_patterns(draft.argv, self.config.allow)
        if covered is None:
            learned = self.rules.lookup(draft.argv)
            if learned is not None:
                covered = shlex.join(learned.argv)
        if covered is not None:
            draft = _Draft(draft.name, RiskTier.EXEC_SAFE, draft.argv, draft.paths)
            return self._allow(
                draft,
                rule=f"allowlist:{covered}",
                reason=f"{shlex.join(draft.argv)} is covered by the allowlist rule {covered!r}",
            )

        return await self._ask(
            draft,
            rule="tier:exec_unsafe",
            reason=(
                f"{shlex.join(draft.argv)} is not on the allowlist, so it is treated as an "
                "arbitrary command"
            ),
        )

    async def _evaluate_write(self, draft: _Draft) -> tuple[PolicyVerdict, PolicyDecision]:
        if draft.outside:
            return await self._ask(
                draft,
                rule="path-jail",
                reason=(
                    f"{', '.join(draft.outside)} resolves outside the workspace "
                    f"{self.ctx.workspace} and always needs your approval"
                ),
            )
        if draft.tier is RiskTier.WRITE_OUTSIDE:
            return await self._ask(
                draft,
                rule="tier:write_outside",
                reason=f"{draft.name} writes outside the workspace and always needs your approval",
            )
        if self.config.mode is PolicyMode.AUTO:
            return self._allow(
                draft,
                rule="mode:auto",
                reason="workspace writes are automatic in auto mode",
            )
        return await self._ask(
            draft,
            rule="tier:write_workspace",
            reason=f"writing with {draft.name} needs your approval",
        )

    def _outside(self, paths: tuple[str, ...]) -> tuple[str, ...]:
        """The paths that escape the jail, as the caller wrote them."""
        outside: list[str] = []
        for path in paths:
            try:
                self.ctx.require_within_workspace(path)
            except ToolError:
                outside.append(path)
        return tuple(outside)

    async def _ask(
        self,
        draft: _Draft,
        *,
        rule: str,
        reason: str,
    ) -> tuple[PolicyVerdict, PolicyDecision]:
        request = ApprovalRequest(
            name=draft.name,
            tier=draft.tier,
            reason=reason,
            argv=draft.argv,
            paths=draft.paths,
        )
        approval = await self.approver.approve(request)

        if approval.decision is Decision.DENY:
            return self._deny(
                draft,
                rule=f"approver:{rule}",
                reason=approval.note or reason,
            )

        if request.needs_typed_confirmation:
            expected = request.confirmation
            if approval.typed.strip() != expected.strip():
                return self._deny(
                    draft,
                    rule=f"confirmation:{rule}",
                    reason=(
                        f"this call has to be typed back to be approved. Expected {expected!r}, "
                        f"got {approval.typed!r}. Nothing was executed."
                    ),
                )

        final_rule = f"approver:{rule}"
        if approval.remember and draft.argv:
            learned = self.rules.remember(draft.argv)
            final_rule = f"learned:{shlex.join(learned.argv)}"
        elif approval.note:
            final_rule = f"approver:{rule} ({approval.note})"

        return self._allow(draft, rule=final_rule, reason=reason)

    def _allow(
        self, draft: _Draft, *, rule: str, reason: str
    ) -> tuple[PolicyVerdict, PolicyDecision]:
        return (
            PolicyVerdict(Decision.ALLOW, reason),
            _record(draft, Decision.ALLOW, rule, reason),
        )

    def _deny(
        self, draft: _Draft, *, rule: str, reason: str
    ) -> tuple[PolicyVerdict, PolicyDecision]:
        return (
            PolicyVerdict(Decision.DENY, reason),
            _record(draft, Decision.DENY, rule, reason),
        )


def _argv_of(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    """The command in ``arguments``, normalised for matching.

    Accepts both shapes the loop can produce: the argv list the ``shell`` tool
    declares, and a plain string, which a model writes often enough that
    refusing it would just be noise. A string is split with shell quoting rules
    and matched as data — it is never executed here.
    """
    command = arguments.get("command")
    if isinstance(command, str):
        return normalise_argv(shlex.split(command))
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        return normalise_argv([str(item) for item in command])
    return ()


def _paths_of(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    path = arguments.get("path")
    return (path,) if isinstance(path, str) and path else ()


def _match_patterns(argv: tuple[str, ...], patterns: tuple[str, ...]) -> str | None:
    """The first pattern that is a prefix of ``argv``, or ``None``."""
    if not argv:
        return None
    for pattern in patterns:
        tokens = normalise_argv(shlex.split(pattern))
        if not tokens:
            continue
        if (
            len(tokens) <= len(argv)
            and tuple(token.lower() for token in argv[: len(tokens)]) == tokens
        ):
            return pattern
    return None


def _record(draft: _Draft, decision: Decision, rule: str, reason: str) -> PolicyDecision:
    return PolicyDecision(
        at=datetime.now(UTC).isoformat(timespec="seconds"),
        name=draft.name,
        tier=draft.tier,
        decision=decision,
        rule=rule,
        reason=reason,
        argv=draft.argv,
        paths=draft.paths,
    )

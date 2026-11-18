"""Tests for the policy engine.

The promise under test is the tier table in PLAN.md, plus the two properties that
make it worth having: every verdict names the rule behind it, and nothing the
engine cannot classify is allowed. Where a verdict depends on a human, these
tests supply an approver and assert what was *asked*, not what a person would say.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import PolicyConfig, PolicyMode
from bevel.core.messages import ToolCall
from bevel.core.policy.approvals import (
    Approval,
    ApprovalRequest,
    Approver,
    AutoApprover,
    Decision,
    DenyAllApprover,
)
from bevel.core.policy.engine import PolicyEngine, PolicyRequest, allow_all_policy
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec


class FakeTool:
    """A tool that declares a tier and does nothing."""

    def __init__(self, tier: RiskTier, name: str = "shell") -> None:
        self.risk_tier = tier
        self.is_parallel_safe = False
        self.spec = ToolSpec(name=name, description="for tests", parameters={})

    async def execute(
        self,
        arguments: Mapping[str, Any],  # noqa: ARG002
        ctx: ToolContext,  # noqa: ARG002
    ) -> ToolResult:
        return ToolResult(ok=True)


class RecordingApprover:
    """Answers every request the same way, and remembers what it was asked."""

    def __init__(self, approval: Approval) -> None:
        self.approval = approval
        self.requests: list[ApprovalRequest] = []

    def approve(self, request: ApprovalRequest) -> Approval:
        self.requests.append(request)
        return self.approval


def _request(
    tool: FakeTool | None,
    name: str = "shell",
    arguments: Mapping[str, Any] | None = None,
) -> PolicyRequest:
    call = ToolCall(id="call-1", name=name, arguments="{}")
    return PolicyRequest(call=call, tool=tool, name=name, arguments=arguments)


def _shell(command: Any) -> PolicyRequest:
    return _request(FakeTool(RiskTier.EXEC_UNSAFE), arguments={"command": command})


def _write(tier: RiskTier = RiskTier.WRITE_WORKSPACE, path: str = "a.py") -> PolicyRequest:
    return _request(FakeTool(tier, "edit"), name="edit", arguments={"path": path})


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


@pytest.fixture
def engines(ctx: ToolContext, tmp_path: Path) -> Iterator[Callable[..., PolicyEngine]]:
    """Build engines whose database lives in ``tmp_path`` and always gets closed."""
    created: list[PolicyEngine] = []

    def make(
        mode: PolicyMode = PolicyMode.AUTO,
        allow: tuple[str, ...] = (),
        deny: tuple[str, ...] = (),
        approver: Approver | None = None,
    ) -> PolicyEngine:
        config = PolicyConfig(mode=mode, allow=allow, deny=deny, db_path=tmp_path / "policy.db")
        engine = PolicyEngine(ctx, config=config, approver=approver)
        created.append(engine)
        return engine

    yield make
    for engine in created:
        engine.close()


class TestTierContract:
    def test_reads_are_free(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(approver=DenyAllApprover())
        verdict = engine(_write(RiskTier.READ, "a.py"))
        assert verdict.decision is Decision.ALLOW
        assert engine.decisions()[0].rule == "tier:read"

    def test_workspace_writes_are_free_in_auto_mode(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(mode=PolicyMode.AUTO, approver=DenyAllApprover())
        assert engine(_write()).decision is Decision.ALLOW
        assert engine.decisions()[0].rule == "mode:auto"

    def test_workspace_writes_ask_in_ask_mode(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(mode=PolicyMode.ASK, approver=DenyAllApprover())
        assert engine(_write()).decision is Decision.DENY
        assert engine.decisions()[0].rule == "approver:tier:write_workspace"

    def test_an_escaping_path_always_asks(self, engines: Callable[..., PolicyEngine]) -> None:
        for mode in PolicyMode:
            engine = engines(mode=mode, approver=DenyAllApprover())
            verdict = engine(_write(path="../out.py"))
            assert verdict.decision is Decision.DENY
            assert engine.decisions()[0].rule == "approver:path-jail"

    def test_a_write_outside_tool_always_asks(self, engines: Callable[..., PolicyEngine]) -> None:
        for mode in PolicyMode:
            engine = engines(mode=mode, approver=DenyAllApprover())
            verdict = engine(_write(RiskTier.WRITE_OUTSIDE))
            assert verdict.decision is Decision.DENY
            assert engine.decisions()[0].rule == "approver:tier:write_outside"

    def test_network_egress_asks(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(mode=PolicyMode.AUTO, approver=DenyAllApprover())
        verdict = engine(_request(FakeTool(RiskTier.NETWORK_EGRESS, "fetch"), arguments={}))
        assert verdict.decision is Decision.DENY
        assert engine.decisions()[0].rule == "approver:tier:network_egress"

    def test_auto_mode_relaxes_workspace_writes_only(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        """Auto mode is not a yolo switch: the always-ask tiers still ask."""
        engine = engines(mode=PolicyMode.AUTO, approver=DenyAllApprover())
        engine(_write(RiskTier.WRITE_OUTSIDE))
        assert engine.decisions()[0].rule == "approver:tier:write_outside"
        engine(_write(path="../out.py"))
        assert engine.decisions()[0].rule == "approver:path-jail"


class TestExecRules:
    def test_an_allowlisted_command_runs_unasked(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        verdict = engine(_shell(["git", "status", "--short"]))
        assert verdict.decision is Decision.ALLOW
        record = engine.decisions()[0]
        assert record.rule == "allowlist:git status"
        assert record.tier is RiskTier.EXEC_SAFE

    def test_a_command_string_is_matched_as_data(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        assert engine(_shell("git status --short")).decision is Decision.ALLOW

    def test_an_unlisted_command_asks(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(approver=DenyAllApprover())
        assert engine(_shell(["pytest", "-q"])).decision is Decision.DENY
        assert engine.decisions()[0].rule == "approver:tier:exec_unsafe"

    def test_a_config_allow_pattern_makes_a_command_free(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(allow=("pytest",), approver=DenyAllApprover())
        assert engine(_shell(["pytest", "-q", "tests/unit"])).decision is Decision.ALLOW
        assert engine.decisions()[0].rule == "allowlist:pytest"

    def test_a_config_deny_pattern_beats_the_allowlist(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(allow=("git",), deny=("git status",), approver=DenyAllApprover())
        assert engine(_shell(["git", "status"])).decision is Decision.DENY
        assert engine.decisions()[0].rule == "config:deny:git status"

    def test_a_deny_pattern_refuses_without_asking(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW))
        engine = engines(deny=("git status",), approver=approver)
        engine(_shell(["git", "status"]))
        assert approver.requests == []

    def test_a_destructive_command_always_asks(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(mode=PolicyMode.AUTO, approver=DenyAllApprover())
        assert engine(_shell(["rm", "-rf", "build"])).decision is Decision.DENY
        record = engine.decisions()[0]
        assert record.rule == "approver:builtin:rm recursive force"
        assert record.tier is RiskTier.EXEC_DESTRUCTIVE

    def test_a_learned_rule_cannot_make_a_destructive_command_free(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine.remember(["git", "reset"])
        assert engine(_shell(["git", "reset", "--hard", "HEAD~1"])).decision is Decision.DENY
        assert engine.decisions()[0].rule == "approver:builtin:git reset --hard"

    def test_a_learned_rule_makes_an_ordinary_command_free(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine.remember(["pytest"])
        assert engine(_shell(["pytest", "-q"])).decision is Decision.ALLOW
        assert engine.decisions()[0].rule == "allowlist:pytest"

    def test_a_broad_learned_rule_covers_a_longer_command(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine.remember(["git"])
        engine(_shell(["git", "commit", "-m", "msg"]))
        assert engine.decisions()[0].rule == "allowlist:git"

    def test_revoking_a_learned_rule_makes_it_ask_again(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine.remember(["pytest"])
        assert engine.revoke(["pytest"]) is True
        assert engine(_shell(["pytest"])).decision is Decision.DENY
        assert engine.revoke(["pytest"]) is False

    def test_a_command_tier_tool_with_no_command_is_refused(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=AutoApprover())
        verdict = engine(_request(FakeTool(RiskTier.EXEC_UNSAFE), arguments={}))
        assert verdict.decision is Decision.DENY
        assert engine.decisions()[0].rule == "exec:unclassifiable"

    def test_a_command_tier_tool_with_a_blank_command_is_refused(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=AutoApprover())
        assert engine(_shell([])).decision is Decision.DENY
        assert engine.decisions()[0].rule == "exec:unclassifiable"

    def test_a_command_that_is_not_a_string_is_refused(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=AutoApprover())
        assert engine(_shell(42)).decision is Decision.DENY
        assert engine.decisions()[0].rule == "exec:unclassifiable"


class TestUnknownTool:
    def test_an_unregistered_tool_is_refused(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(approver=AutoApprover())
        verdict = engine(_request(None, name="mystery", arguments={"command": "ls"}))
        assert verdict.decision is Decision.DENY
        assert engine.decisions()[0].rule == "unknown-tool"
        assert "not a registered tool" in verdict.reason

    def test_the_engine_never_asks_about_a_tool_it_cannot_classify(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW))
        engines(approver=approver)(_request(None, name="mystery"))
        assert approver.requests == []


class TestTypedConfirmation:
    def test_a_destructive_call_needs_the_command_typed_back(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW))
        engine = engines(approver=approver)
        verdict = engine(_shell(["rm", "-rf", "build"]))
        assert verdict.decision is Decision.DENY
        assert engine.decisions()[0].rule == "confirmation:builtin:rm recursive force"
        assert "'rm -rf build'" in verdict.reason

    def test_typing_it_back_is_enough(self, engines: Callable[..., PolicyEngine]) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW, typed="rm -rf build"))
        engine = engines(approver=approver)
        assert engine(_shell(["rm", "-rf", "build"])).decision is Decision.ALLOW

    def test_an_ordinary_call_needs_no_typing(self, engines: Callable[..., PolicyEngine]) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW))
        engine = engines(approver=approver)
        assert engine(_shell(["pytest", "-q"])).decision is Decision.ALLOW
        assert approver.requests[0].needs_typed_confirmation is False

    def test_typing_the_wrong_thing_does_not_approve_it(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW, typed="rm -rf /"))
        engine = engines(approver=approver)
        assert engine(_shell(["rm", "-rf", "build"])).decision is Decision.DENY


class TestRememberFromAnApproval:
    def test_an_approval_can_learn_the_command(self, engines: Callable[..., PolicyEngine]) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW, remember=True))
        engine = engines(approver=approver)
        assert engine(_shell(["pytest", "-q"])).decision is Decision.ALLOW
        assert engine.decisions()[0].rule == "learned:pytest -q"

    def test_a_learned_approval_makes_the_next_call_free(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=RecordingApprover(Approval(Decision.ALLOW, remember=True)))
        engine(_shell(["pytest", "-q"]))
        engine.approver = DenyAllApprover()
        assert engine(_shell(["pytest", "-q", "tests/unit"])).decision is Decision.ALLOW

    def test_a_deny_never_learns_anything(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(approver=DenyAllApprover())
        engine(_shell(["pytest"]))
        assert engine.rules.entries() == ()

    def test_remembering_a_call_with_no_command_persists_nothing(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.ALLOW, remember=True))
        engine = engines(mode=PolicyMode.ASK, approver=approver)
        assert engine(_write()).decision is Decision.ALLOW
        assert engine.rules.entries() == ()
        assert engine.decisions()[0].rule == "approver:tier:write_workspace"


class TestApproverSeam:
    def test_the_request_carries_the_facts_a_human_needs(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.DENY))
        engine = engines(mode=PolicyMode.ASK, approver=approver)
        engine(_write(path="notes to self.md"))
        request = approver.requests[0]
        assert request.name == "edit"
        assert request.tier is RiskTier.WRITE_WORKSPACE
        assert request.paths == ("notes to self.md",)
        assert request.reason

    def test_the_command_is_passed_to_the_approver(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        approver = RecordingApprover(Approval(Decision.DENY))
        engines(approver=approver)(_shell(["pytest", "-q"]))
        assert approver.requests[0].argv == ("pytest", "-q")

    def test_an_approver_deny_explains_itself(self, engines: Callable[..., PolicyEngine]) -> None:
        approver = RecordingApprover(Approval(Decision.DENY, note="you said no"))
        engine = engines(approver=approver)
        verdict = engine(_shell(["pytest"]))
        assert verdict.reason == "you said no"
        assert engine.decisions()[0].rule == "approver:tier:exec_unsafe"

    def test_the_headless_default_refuses_rather_than_guessing(
        self, ctx: ToolContext, tmp_path: Path
    ) -> None:
        engine = PolicyEngine(ctx, config=PolicyConfig(db_path=tmp_path / "policy.db"))
        try:
            assert isinstance(engine.approver, DenyAllApprover)
            assert engine(_shell(["pytest"])).decision is Decision.DENY
        finally:
            engine.close()


class TestDecisionLog:
    def test_every_call_is_logged_with_the_rule_that_fired(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine(_write(RiskTier.READ, "a.py"))
        engine(_shell(["pytest"]))
        records = engine.decisions()
        assert [record.rule for record in records] == ["approver:tier:exec_unsafe", "tier:read"]
        assert [record.decision for record in records] == [Decision.DENY, Decision.ALLOW]

    def test_the_log_round_trips_argv_and_paths(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(mode=PolicyMode.ASK, approver=DenyAllApprover())
        engine(_shell(["git", "commit", "-m", "two words"]))
        engine(_write(path="a file.txt"))
        write_record, exec_record = engine.decisions(2)
        assert exec_record.argv == ("git", "commit", "-m", "two words")
        assert write_record.paths == ("a file.txt",)
        assert write_record.name == "edit"
        assert write_record.at

    def test_the_log_honours_the_limit(self, engines: Callable[..., PolicyEngine]) -> None:
        engine = engines(approver=DenyAllApprover())
        for _ in range(4):
            engine(_shell(["pytest"]))
        assert len(engine.decisions(limit=2)) == 2

    def test_the_log_survives_a_reopened_engine(self, ctx: ToolContext, tmp_path: Path) -> None:
        db = tmp_path / "policy.db"
        first = PolicyEngine(ctx, config=PolicyConfig(db_path=db))
        first(_shell(["git", "status"]))
        first.close()
        second = PolicyEngine(ctx, config=PolicyConfig(db_path=db))
        try:
            assert second.decisions()[0].rule == "allowlist:git status"
        finally:
            second.close()

    def test_an_engine_that_never_decides_leaves_no_database(
        self, ctx: ToolContext, tmp_path: Path
    ) -> None:
        db = tmp_path / "nested" / "policy.db"
        PolicyEngine(ctx, config=PolicyConfig(db_path=db)).close()
        assert not db.exists()

    def test_a_denial_is_logged_with_the_tier_it_was_classified_as(
        self, engines: Callable[..., PolicyEngine]
    ) -> None:
        engine = engines(approver=DenyAllApprover())
        engine(_shell(["rm", "-rf", "build"]))
        record = engine.decisions()[0]
        assert record.tier is RiskTier.EXEC_DESTRUCTIVE
        assert record.name == "shell"


def test_allow_all_policy_is_the_caller_s_opt_out_not_the_default() -> None:
    """The engine is the default; a caller with its own isolation can still opt out."""
    call = ToolCall(id="call-1", name="shell", arguments="{}")
    request = PolicyRequest(call=call, tool=None, name="shell", arguments={"command": "rm -rf /"})
    assert allow_all_policy(request).decision is Decision.ALLOW

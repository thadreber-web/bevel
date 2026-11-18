"""Tests for the approval seam.

The engine, not the approver, decides whether an answer is good enough — so
these tests pin the shape of the question and the honesty of the two headless
answers. The m07 TUI modal will be a third implementation of the same protocol.
"""

from __future__ import annotations

import pytest

from bevel.core.policy.approvals import (
    Approval,
    ApprovalRequest,
    Approver,
    AutoApprover,
    Decision,
    DenyAllApprover,
)
from bevel.core.tools.base import RiskTier


def _request(tier: RiskTier = RiskTier.EXEC_UNSAFE, argv: tuple[str, ...] = ()) -> ApprovalRequest:
    return ApprovalRequest(name="shell", tier=tier, reason="needs approval", argv=argv)


class TestApprovalRequest:
    def test_only_destructive_calls_need_typing(self) -> None:
        for tier in RiskTier:
            request = _request(tier)
            assert request.needs_typed_confirmation is (tier is RiskTier.EXEC_DESTRUCTIVE)

    def test_confirmation_is_the_command(self) -> None:
        request = _request(RiskTier.EXEC_DESTRUCTIVE, ("rm", "-rf", "build"))
        assert request.confirmation == "rm -rf build"

    def test_confirmation_falls_back_to_the_tool_name(self) -> None:
        request = _request(RiskTier.EXEC_DESTRUCTIVE)
        assert request.confirmation == "shell"

    def test_an_answer_carries_no_decisions_of_its_own(self) -> None:
        approval = Approval(Decision.ALLOW)
        assert approval.note == ""
        assert approval.typed == ""
        assert approval.remember is False


class TestDenyAllApprover:
    def test_everything_is_refused(self) -> None:
        for tier in RiskTier:
            approval = DenyAllApprover().approve(_request(tier))
            assert approval.decision is Decision.DENY

    def test_the_note_says_nobody_was_asked(self) -> None:
        approval = DenyAllApprover().approve(_request(RiskTier.WRITE_OUTSIDE))
        assert "no approver is installed" in approval.note
        assert "shell" in approval.note
        assert "write_outside" in approval.note

    def test_the_note_names_the_command_so_the_model_can_ask(self) -> None:
        approval = DenyAllApprover().approve(_request(argv=("git", "push")))
        assert "'git push'" in approval.note

    def test_a_deny_carries_no_typed_text(self) -> None:
        approval = DenyAllApprover().approve(
            _request(RiskTier.EXEC_DESTRUCTIVE, ("rm", "-rf", "x"))
        )
        assert approval.typed == ""


class TestAutoApprover:
    def test_everything_is_allowed(self) -> None:
        for tier in RiskTier:
            approval = AutoApprover().approve(_request(tier))
            assert approval.decision is Decision.ALLOW

    def test_the_log_can_tell_it_was_not_a_human(self) -> None:
        approval = AutoApprover().approve(_request(RiskTier.WRITE_OUTSIDE))
        assert "no human was consulted" in approval.note

    def test_it_supplies_the_confirmation_text(self) -> None:
        request = _request(RiskTier.EXEC_DESTRUCTIVE, ("git", "clean", "-fd"))
        approval = AutoApprover().approve(request)
        assert approval.typed == request.confirmation

    def test_it_does_not_learn_rules_on_the_operator_s_behalf(self) -> None:
        assert AutoApprover().approve(_request()).remember is False


class TestApproverProtocol:
    def test_a_hand_written_approver_is_all_the_seam_needs(self) -> None:
        class Recorder:
            def __init__(self) -> None:
                self.seen: list[ApprovalRequest] = []

            def approve(self, request: ApprovalRequest) -> Approval:
                self.seen.append(request)
                return Approval(Decision.ALLOW, note="recorded")

        recorder = Recorder()
        approval = recorder.approve(_request(RiskTier.EXEC_UNSAFE, ("pytest",)))
        assert approval.decision is Decision.ALLOW
        assert recorder.seen[0].argv == ("pytest",)

    @pytest.mark.parametrize("approver", [DenyAllApprover(), AutoApprover()])
    def test_builtin_approvers_answer_a_request_without_asking_for_more(
        self, approver: Approver
    ) -> None:
        assert isinstance(approver.approve(_request()), Approval)

"""Tests for the tool seam: the path jail, the result type, and dispatch.

The seam is what makes the six tools behave as one surface, so these tests
target the promises the seam owns -- arguments are validated before anything
runs, nothing raises into the loop, and a path cannot leave the workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bevel.core.tools.base import (
    RiskTier,
    ToolContext,
    ToolError,
    ToolResult,
    call_tool,
)
from bevel.core.tools.types import ToolSpec

ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "shout": {"type": "boolean"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


class FakeTool:
    """A tool whose body is supplied by the test."""

    spec: ToolSpec = ToolSpec(name="fake", description="for tests", parameters=ARGS_SCHEMA)
    risk_tier: RiskTier = RiskTier.READ
    is_parallel_safe: bool = True

    def __init__(self, body: Any = None) -> None:
        self.seen: list[dict[str, Any]] = []
        self._body = body

    async def execute(self, arguments: Any, ctx: ToolContext) -> ToolResult:
        self.seen.append(dict(arguments))
        if self._body is not None:
            return await self._body(arguments, ctx)
        return ToolResult(ok=True, content=arguments["path"])


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


class TestToolResult:
    def test_failure_is_a_result_not_an_exception(self) -> None:
        result = ToolResult.failure("boom")
        assert result.ok is False
        assert result.failed is True
        assert result.error == "boom"
        assert result.content == ""

    def test_ok_result_carries_no_error(self) -> None:
        result = ToolResult(ok=True, content="hi")
        assert result.failed is False
        assert result.error is None

    def test_result_is_hashable_so_it_can_be_cached(self) -> None:
        assert hash(ToolResult(ok=True, content="a")) == hash(ToolResult(ok=True, content="a"))


class TestToolContext:
    def test_relative_paths_resolve_against_the_workspace(self, ctx: ToolContext) -> None:
        assert ctx.resolve("pkg/app.py") == ctx.workspace / "pkg" / "app.py"

    def test_backslashes_are_accepted_on_any_platform(self, ctx: ToolContext) -> None:
        assert ctx.resolve(r"pkg\app.py") == ctx.workspace / "pkg" / "app.py"

    def test_tilde_is_expanded(self, tmp_path: Path) -> None:
        context = ToolContext(workspace=tmp_path)
        resolved = context.resolve("~")
        assert resolved.is_absolute()
        assert resolved != context.workspace

    def test_workspace_is_canonicalised_once(self, tmp_path: Path) -> None:
        context = ToolContext(workspace=Path(str(tmp_path), "."))
        assert context.workspace == tmp_path.resolve()

    def test_traversal_is_refused_and_names_the_milestone(self, ctx: ToolContext) -> None:
        with pytest.raises(ToolError) as excinfo:
            ctx.require_within_workspace("../secrets.env")
        assert "outside the workspace" in str(excinfo.value)
        assert "m05" in str(excinfo.value)

    def test_traversal_that_returns_inside_is_allowed(self, ctx: ToolContext) -> None:
        allowed = ctx.require_within_workspace("pkg/../pkg/app.py")
        assert allowed == ctx.workspace / "pkg" / "app.py"

    def test_absolute_path_inside_workspace_is_allowed(self, ctx: ToolContext) -> None:
        target = ctx.workspace / "a.py"
        assert ctx.require_within_workspace(str(target)) == target

    def test_absolute_path_outside_is_refused(
        self, ctx: ToolContext, tmp_path_factory: Any
    ) -> None:
        other = tmp_path_factory.mktemp("elsewhere") / "a.py"
        with pytest.raises(ToolError):
            ctx.require_within_workspace(str(other))

    def test_case_difference_does_not_escape_on_windows(self, ctx: ToolContext) -> None:
        inside = str(ctx.workspace / "a.py").upper()
        assert ctx.require_within_workspace(inside) == ctx.workspace / "a.py"

    def test_display_is_relative_inside_and_absolute_outside(self, ctx: ToolContext) -> None:
        assert ctx.display(ctx.workspace / "pkg" / "app.py") == "pkg/app.py"
        assert ctx.display(Path("/somewhere/else.py")) == str(Path("/somewhere/else.py"))

    def test_relative_uses_forward_slashes(self, ctx: ToolContext) -> None:
        assert ctx.relative(ctx.workspace / "pkg" / "app.py") == "pkg/app.py"


class TestCallTool:
    async def test_valid_arguments_reach_the_tool(self, ctx: ToolContext) -> None:
        tool = FakeTool()
        result = await call_tool(tool, {"path": "a.py"}, ctx)
        assert result.ok is True
        assert tool.seen == [{"path": "a.py"}]

    async def test_invalid_arguments_never_reach_the_tool(self, ctx: ToolContext) -> None:
        tool = FakeTool()
        result = await call_tool(tool, {"path": 7}, ctx)
        assert result.failed
        assert tool.seen == []
        assert "validation" in (result.error or "")
        assert "$.path" in (result.error or "")

    async def test_every_violation_is_reported_in_one_message(self, ctx: ToolContext) -> None:
        result = await call_tool(FakeTool(), {"shout": "yes", "extra": 1}, ctx)
        error = result.error or ""
        assert "$.path" in error
        assert "$.shout" in error
        assert "$.extra" in error
        assert error.startswith("fake:")

    async def test_tool_error_becomes_a_failure_naming_the_tool(self, ctx: ToolContext) -> None:
        async def boom(_arguments: Any, _ctx: ToolContext) -> ToolResult:
            raise ToolError("no such file")

        result = await call_tool(FakeTool(boom), {"path": "a.py"}, ctx)
        assert result.failed
        assert result.error == "fake: no such file"

    async def test_oserror_keeps_its_type_so_the_model_can_tell_them_apart(
        self, ctx: ToolContext
    ) -> None:
        async def boom(_arguments: Any, _ctx: ToolContext) -> ToolResult:
            raise FileNotFoundError(2, "No such file or directory")

        result = await call_tool(FakeTool(boom), {"path": "a.py"}, ctx)
        assert result.failed
        assert "FileNotFoundError" in (result.error or "")

    async def test_a_path_jail_refusal_surfaces_as_a_failure(self, ctx: ToolContext) -> None:
        async def leak(arguments: Any, context: ToolContext) -> ToolResult:
            context.require_within_workspace(arguments["path"])
            return ToolResult(ok=True)

        result = await call_tool(FakeTool(leak), {"path": "../../etc/passwd"}, ctx)
        assert result.failed
        assert "outside the workspace" in (result.error or "")

    def test_the_seven_tiers_the_plan_names_are_all_declared(self) -> None:
        assert {tier.value for tier in RiskTier} == {
            "read",
            "write_workspace",
            "write_outside",
            "exec_safe",
            "exec_unsafe",
            "exec_destructive",
            "network_egress",
        }

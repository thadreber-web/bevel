"""Tests for ``grep`` and ``glob``.

Both are pure Python on purpose -- a bundled ripgrep binary would be a compiled
dependency, which the three-dependency rule forbids -- so these tests pin the
behaviour a model actually relies on: paths it can feed straight back into
``read``, patterns that do not cross directory boundaries, and a result set that
is capped and stable rather than unbounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bevel.core.tools.base import RiskTier, ToolContext, ToolResult, call_tool
from bevel.core.tools.glob import GlobTool
from bevel.core.tools.grep import GrepTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


def make(ctx: ToolContext, rel: str, text: str) -> Path:
    path = ctx.workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


async def grep(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(GrepTool(), arguments, ctx)


async def glob(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(GlobTool(), arguments, ctx)


TREE = {
    "src/app.py": ("import os\n\ndef handler(request):\n    return os.getcwd()\n\n# end\n"),
    "src/util/helper.py": "HANDLER = 'upper'\n",
    "src/app_test.py": "def test_handler():\n    assert handler\n",
    "README.md": "# handler\n",
}


@pytest.fixture
def tree(ctx: ToolContext) -> ToolContext:
    for rel, text in TREE.items():
        make(ctx, rel, text)
    return ctx


class TestDeclarations:
    def test_both_are_free_and_parallel_safe(self) -> None:
        for tool in (GrepTool(), GlobTool()):
            assert tool.risk_tier is RiskTier.READ
            assert tool.is_parallel_safe is True

    def test_grep_requires_only_a_pattern(self) -> None:
        assert GrepTool().spec.parameters["required"] == ["pattern"]

    def test_glob_requires_only_a_pattern(self) -> None:
        assert GlobTool().spec.parameters["required"] == ["pattern"]


class TestGrep:
    async def test_matches_are_reported_as_path_line_text(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="def handler")
        assert result.ok is True
        assert result.content.splitlines() == ["src/app.py:3:def handler(request):"]

    async def test_paths_are_workspace_relative_with_forward_slashes(
        self, tree: ToolContext
    ) -> None:
        result = await grep(tree, pattern="HANDLER")
        assert result.content.splitlines()[0].startswith("src/util/helper.py:1:")

    async def test_regex_works_and_is_not_escaped_into_literal(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern=r"def \w+\(")
        assert "src/app.py:3:" in result.content
        assert "src/app_test.py:1:" in result.content

    async def test_invalid_regex_is_reported_not_swallowed(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="def (")
        assert result.failed
        assert "pattern" in (result.error or "")

    async def test_ignore_case(self, tree: ToolContext) -> None:
        sensitive = await grep(tree, pattern="handler")
        insensitive = await grep(tree, pattern="handler", ignore_case=True)
        assert insensitive.content.count("\n") >= sensitive.content.count("\n")
        assert "README.md:1:# handler" in insensitive.content
        assert "src/util/helper.py:1:HANDLER = 'upper'" in insensitive.content

    async def test_glob_filter_selects_files(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="handler", glob="*_test.py")
        assert result.content.splitlines() == [
            "src/app_test.py:1:def test_handler():",
            "src/app_test.py:2:    assert handler",
        ]

    async def test_glob_filter_can_target_a_directory(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="HANDLER", glob="util/*.py")
        assert result.content.splitlines() == ["src/util/helper.py:1:HANDLER = 'upper'"]

    async def test_search_can_be_scoped_to_a_subdirectory(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="HANDLER", path="src/util")
        assert result.content.splitlines() == ["src/util/helper.py:1:HANDLER = 'upper'"]

    async def test_context_lines_surround_a_match(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="return os.getcwd", context=2)
        lines = result.content.splitlines()
        assert lines[0] == "src/app.py-2-"
        assert "src/app.py:4:    return os.getcwd()" in lines
        assert lines[-1] == "src/app.py-6-# end"

    async def test_results_are_capped_and_says_so(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern=".", max_results=2)
        assert len([line for line in result.content.splitlines() if ":" in line]) <= 2
        assert "max_results" in result.content

    async def test_no_matches_is_a_success_with_an_explicit_note(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="nothing like this")
        assert result.ok is True
        assert "no match" in result.content.lower()

    async def test_binary_files_are_skipped_and_counted(self, tree: ToolContext) -> None:
        make(tree, "blob.dat", "\x00\x00handler\x00")
        result = await grep(tree, pattern="handler")
        assert result.ok is True
        assert "binary" in result.content

    async def test_junk_directories_are_never_searched(self, tree: ToolContext) -> None:
        make(tree, "node_modules/pkg/index.py", "handler = 1\n")
        make(tree, ".git/hooks/app.py", "handler = 1\n")
        make(tree, "src/__pycache__/app.py", "handler = 1\n")
        result = await grep(tree, pattern="handler")
        assert "node_modules" not in result.content
        assert ".git" not in result.content
        assert "__pycache__" not in result.content

    async def test_paths_outside_the_workspace_are_refused(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="x", path="../")
        assert result.failed
        assert "outside the workspace" in (result.error or "")

    async def test_missing_search_root_is_a_failure(self, tree: ToolContext) -> None:
        result = await grep(tree, pattern="x", path="nope")
        assert result.failed


class TestGlob:
    async def test_bare_pattern_matches_basenames_anywhere(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="*.py")
        assert result.content.splitlines() == [
            "src/app.py",
            "src/app_test.py",
            "src/util/helper.py",
        ]

    async def test_pattern_with_a_separator_is_anchored_to_the_root(
        self, tree: ToolContext
    ) -> None:
        result = await glob(tree, pattern="src/*.py")
        assert result.content.splitlines() == ["src/app.py", "src/app_test.py"]

    async def test_double_star_crosses_directories(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="src/**/*.py")
        assert "src/util/helper.py" in result.content

    async def test_question_mark_matches_one_character(self, tree: ToolContext) -> None:
        make(tree, "src/a1.py", "")
        result = await glob(tree, pattern="src/a?.py")
        assert result.content.splitlines() == ["src/a1.py"]

    async def test_results_are_sorted(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="**/*.py")
        lines = result.content.splitlines()
        assert lines == sorted(lines)

    async def test_no_matches_is_explicit(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="*.rs")
        assert result.ok is True
        assert "no match" in result.content.lower()

    async def test_results_are_capped_and_says_so(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="**/*.py", max_results=2)
        assert len(result.content.splitlines()) <= 3
        assert "max_results" in result.content

    async def test_scoped_to_a_subdirectory(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="*.py", path="src/util")
        assert result.content.splitlines() == ["src/util/helper.py"]

    async def test_junk_directories_are_never_listed(self, tree: ToolContext) -> None:
        make(tree, "node_modules/pkg/index.py", "")
        result = await glob(tree, pattern="**/*.py")
        assert "node_modules" not in result.content

    async def test_paths_outside_the_workspace_are_refused(self, tree: ToolContext) -> None:
        result = await glob(tree, pattern="*", path="../")
        assert result.failed
        assert "outside the workspace" in (result.error or "")

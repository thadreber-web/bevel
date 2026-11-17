"""Tests for ``write`` and ``edit``.

``edit`` is the tool the whole milestone is judged on: it has to apply a real
diff, refuse an ambiguous match instead of guessing, and leave the file intact
when one edit out of five does not apply. Line-ending preservation matters
because this project is developed on Windows, where a tool that rewrites every
CRLF to LF produces a diff that touches the entire file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bevel.core.tools.base import RiskTier, ToolContext, ToolResult, call_tool
from bevel.core.tools.edit import EditTool
from bevel.core.tools.write import WriteTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


async def write(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(WriteTool(), arguments, ctx)


async def edit(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(EditTool(), arguments, ctx)


def make(ctx: ToolContext, rel: str, data: str | bytes) -> Path:
    path = ctx.workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


class TestDeclarations:
    def test_both_are_workspace_writes_and_not_parallel_safe(self) -> None:
        for tool in (WriteTool(), EditTool()):
            assert tool.risk_tier is RiskTier.WRITE_WORKSPACE
            assert tool.is_parallel_safe is False

    def test_write_requires_path_and_content(self) -> None:
        assert WriteTool().spec.parameters["required"] == ["path", "content"]

    def test_edit_requires_path_and_edits(self) -> None:
        assert EditTool().spec.parameters["required"] == ["path", "edits"]


class TestWrite:
    async def test_creates_a_new_file_and_parent_directories(self, ctx: ToolContext) -> None:
        result = await write(ctx, path="pkg/deep/new.py", content="x = 1\n")
        assert result.ok is True
        assert (ctx.workspace / "pkg" / "deep" / "new.py").read_text() == "x = 1\n"

    async def test_refuses_to_clobber_without_saying_so(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "original\n")
        result = await write(ctx, path="a.py", content="replacement\n")
        assert result.failed
        assert "overwrite" in (result.error or "")
        assert read_bytes(ctx.workspace / "a.py") == b"original\n"

    async def test_overwrite_replaces_and_reports_the_diff(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "one\ntwo\n")
        result = await write(ctx, path="a.py", content="one\nTWO\n", overwrite=True)
        assert result.ok is True
        assert "-two" in result.content
        assert "+TWO" in result.content
        assert (ctx.workspace / "a.py").read_text() == "one\nTWO\n"

    async def test_writing_identical_content_reports_no_change(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "same\n")
        result = await write(ctx, path="a.py", content="same\n", overwrite=True)
        assert result.ok is True
        assert "no change" in result.content.lower()
        assert read_bytes(ctx.workspace / "a.py") == b"same\n"

    async def test_content_is_written_byte_for_byte(self, ctx: ToolContext) -> None:
        await write(ctx, path="a.txt", content="trailing space  \n\ttab\n")
        assert read_bytes(ctx.workspace / "a.txt") == b"trailing space  \n\ttab\n"

    async def test_no_temporary_files_are_left_behind(self, ctx: ToolContext) -> None:
        await write(ctx, path="a.py", content="x = 1\n")
        assert [p.name for p in ctx.workspace.iterdir()] == ["a.py"]

    async def test_paths_outside_the_workspace_are_refused(self, ctx: ToolContext) -> None:
        result = await write(ctx, path="../escape.py", content="x")
        assert result.failed
        assert "outside the workspace" in (result.error or "")
        assert not (ctx.workspace.parent / "escape.py").exists()

    async def test_writing_over_a_directory_is_a_failure(self, ctx: ToolContext) -> None:
        (ctx.workspace / "pkg").mkdir()
        result = await write(ctx, path="pkg", content="x", overwrite=True)
        assert result.failed

    async def test_empty_content_is_allowed(self, ctx: ToolContext) -> None:
        result = await write(ctx, path="empty.txt", content="")
        assert result.ok is True
        assert (ctx.workspace / "empty.txt").read_bytes() == b""


class TestEdit:
    async def test_applies_a_single_replacement(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "def f():\n    return 1\n")
        result = await edit(
            ctx, path="a.py", edits=[{"old_text": "return 1", "new_text": "return 2"}]
        )
        assert result.ok is True
        assert (ctx.workspace / "a.py").read_text() == "def f():\n    return 2\n"

    async def test_returns_a_real_diff(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "one\ntwo\nthree\n")
        result = await edit(ctx, path="a.py", edits=[{"old_text": "two", "new_text": "TWO"}])
        assert result.ok is True
        assert "--- a.py" in result.content
        assert "+++ a.py" in result.content
        assert "-two" in result.content
        assert "+TWO" in result.content
        assert "@@" in result.content

    async def test_multiple_edits_apply_in_order(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "alpha\nbeta\ngamma\n")
        result = await edit(
            ctx,
            path="a.py",
            edits=[
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "gamma", "new_text": "GAMMA"},
            ],
        )
        assert result.ok is True
        assert (ctx.workspace / "a.py").read_text() == "ALPHA\nbeta\nGAMMA\n"
        assert result.content.count("-alpha") == 1

    async def test_an_edit_can_build_on_the_previous_one(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "a\n")
        result = await edit(
            ctx,
            path="a.py",
            edits=[
                {"old_text": "a", "new_text": "ab"},
                {"old_text": "ab", "new_text": "abc"},
            ],
        )
        assert result.ok is True
        assert (ctx.workspace / "a.py").read_text() == "abc\n"

    async def test_ambiguous_match_is_refused_and_names_the_count(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x = 1\nx = 1\nx = 1\n")
        result = await edit(ctx, path="a.py", edits=[{"old_text": "x = 1", "new_text": "x = 2"}])
        assert result.failed
        assert "3 times" in (result.error or "")
        assert "replace_all" in (result.error or "")
        assert read_bytes(ctx.workspace / "a.py") == b"x = 1\nx = 1\nx = 1\n"

    async def test_replace_all_replaces_every_occurrence(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x = 1\nx = 1\n")
        result = await edit(
            ctx,
            path="a.py",
            edits=[{"old_text": "x = 1", "new_text": "x = 2", "replace_all": True}],
        )
        assert result.ok is True
        assert (ctx.workspace / "a.py").read_text() == "x = 2\nx = 2\n"

    async def test_text_that_is_not_present_is_a_failure(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "real\n")
        result = await edit(ctx, path="a.py", edits=[{"old_text": "imagined", "new_text": "x"}])
        assert result.failed
        assert "not found" in (result.error or "")

    async def test_a_failing_edit_rolls_the_whole_call_back(self, ctx: ToolContext) -> None:
        """All-or-nothing: a half-applied edit is worse than none."""
        original = "alpha\nbeta\n"
        make(ctx, "a.py", original)
        result = await edit(
            ctx,
            path="a.py",
            edits=[
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "not there", "new_text": "x"},
            ],
        )
        assert result.failed
        assert read_bytes(ctx.workspace / "a.py") == original.encode()

    async def test_editing_a_missing_file_is_a_failure(self, ctx: ToolContext) -> None:
        result = await edit(ctx, path="nope.py", edits=[{"old_text": "a", "new_text": "b"}])
        assert result.failed
        assert "no such file" in (result.error or "").lower()

    async def test_an_empty_edit_list_is_refused(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x\n")
        result = await edit(ctx, path="a.py", edits=[])
        assert result.failed

    async def test_an_empty_old_text_is_refused(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x\n")
        result = await edit(ctx, path="a.py", edits=[{"old_text": "", "new_text": "y"}])
        assert result.failed
        assert "empty" in (result.error or "").lower()

    async def test_an_edit_that_changes_nothing_is_reported_as_such(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x\n")
        result = await edit(ctx, path="a.py", edits=[{"old_text": "x", "new_text": "x"}])
        assert result.failed
        assert "no change" in (result.error or "").lower()

    async def test_crlf_files_keep_their_crlf(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", b"one\r\ntwo\r\n")
        result = await edit(ctx, path="a.txt", edits=[{"old_text": "two", "new_text": "TWO"}])
        assert result.ok is True
        assert read_bytes(ctx.workspace / "a.txt") == b"one\r\nTWO\r\n"

    async def test_lf_files_are_not_converted_to_crlf(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", b"one\ntwo\n")
        await edit(ctx, path="a.txt", edits=[{"old_text": "two", "new_text": "TWO"}])
        assert read_bytes(ctx.workspace / "a.txt") == b"one\nTWO\n"

    async def test_missing_trailing_newline_is_preserved(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", b"one\ntwo")
        await edit(ctx, path="a.txt", edits=[{"old_text": "two", "new_text": "TWO"}])
        assert read_bytes(ctx.workspace / "a.txt") == b"one\nTWO"

    async def test_a_newline_in_the_replacement_is_written_as_a_new_line(
        self, ctx: ToolContext
    ) -> None:
        make(ctx, "a.txt", b"one\n")
        await edit(ctx, path="a.txt", edits=[{"old_text": "one", "new_text": "one\ntwo"}])
        assert read_bytes(ctx.workspace / "a.txt") == b"one\ntwo\n"

    async def test_crlf_file_gets_crlf_from_a_lf_replacement(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", b"one\r\n")
        await edit(ctx, path="a.txt", edits=[{"old_text": "one", "new_text": "one\ntwo"}])
        assert read_bytes(ctx.workspace / "a.txt") == b"one\r\ntwo\r\n"

    async def test_binary_files_are_refused(self, ctx: ToolContext) -> None:
        make(ctx, "blob.bin", b"\x00\x01\x02")
        result = await edit(ctx, path="blob.bin", edits=[{"old_text": "a", "new_text": "b"}])
        assert result.failed
        assert "binary" in (result.error or "")

    async def test_paths_outside_the_workspace_are_refused(self, ctx: ToolContext) -> None:
        result = await edit(ctx, path="../escape.py", edits=[{"old_text": "a", "new_text": "b"}])
        assert result.failed
        assert "outside the workspace" in (result.error or "")

    async def test_no_temporary_files_are_left_behind(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "x\n")
        await edit(ctx, path="a.py", edits=[{"old_text": "x", "new_text": "y"}])
        assert [p.name for p in ctx.workspace.iterdir()] == ["a.py"]

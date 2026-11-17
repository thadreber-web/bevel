"""Tests for the ``read`` tool.

``read`` is the tool a model uses most, so the tests care about the things that
make its output trustworthy: line numbers that can be quoted back, an honest
answer when the file is binary or missing, and no whole-file slurp of a
multi-gigabyte log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bevel.core.tools.base import RiskTier, ToolContext, ToolResult, call_tool
from bevel.core.tools.read import ReadTool


async def run(ctx: ToolContext, **arguments: Any) -> ToolResult:
    return await call_tool(ReadTool(), arguments, ctx)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


def make(ctx: ToolContext, rel: str, text: str) -> Path:
    path = ctx.workspace / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


class TestDeclaration:
    def test_read_is_free_and_parallel_safe(self) -> None:
        tool = ReadTool()
        assert tool.risk_tier is RiskTier.READ
        assert tool.is_parallel_safe is True
        assert tool.spec.name == "read"

    def test_only_path_is_required(self) -> None:
        assert ReadTool().spec.parameters["required"] == ["path"]


class TestReading:
    async def test_lines_are_numbered_so_they_can_be_quoted_back(self, ctx: ToolContext) -> None:
        make(ctx, "a.py", "first\nsecond\n")
        result = await run(ctx, path="a.py")
        assert result.ok is True
        assert result.content.splitlines() == ["1|first", "2|second"]

    async def test_missing_file_is_a_failure_that_names_the_path(self, ctx: ToolContext) -> None:
        result = await run(ctx, path="nope.py")
        assert result.failed
        assert "nope.py" in (result.error or "")
        assert "no such file" in (result.error or "").lower()

    async def test_a_directory_is_refused_rather_than_dumped(self, ctx: ToolContext) -> None:
        (ctx.workspace / "pkg").mkdir()
        result = await run(ctx, path="pkg")
        assert result.failed
        assert "directory" in (result.error or "")

    async def test_empty_file_reports_itself_as_empty(self, ctx: ToolContext) -> None:
        make(ctx, "empty.txt", "")
        result = await run(ctx, path="empty.txt")
        assert result.ok is True
        assert "empty" in result.content

    async def test_crlf_and_bom_survive_without_stray_carriage_returns(
        self, ctx: ToolContext
    ) -> None:
        make(ctx, "crlf.txt", "\ufeffone\r\ntwo\r\n")
        result = await run(ctx, path="crlf.txt")
        assert result.content.splitlines() == ["1|one", "2|two"]

    async def test_undecodable_bytes_are_replaced_not_fatal(self, ctx: ToolContext) -> None:
        path = ctx.workspace / "latin.txt"
        path.write_bytes(b"caf\xe9\nnext\n")
        result = await call_tool(ReadTool(), {"path": "latin.txt"}, ctx)
        assert result.ok is True
        assert "next" in result.content

    async def test_paths_outside_the_workspace_are_refused(self, ctx: ToolContext) -> None:
        result = await run(ctx, path="../../windows/win.ini")
        assert result.failed
        assert "outside the workspace" in (result.error or "")


class TestOffsets:
    async def test_offset_starts_numbering_at_the_real_line(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", "\n".join(f"line{i}" for i in range(1, 11)))
        result = await run(ctx, path="a.txt", offset=8, limit=2)
        assert result.content.splitlines()[:2] == ["8|line8", "9|line9"]

    async def test_more_lines_are_announced_with_the_next_offset(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", "\n".join(f"line{i}" for i in range(1, 11)))
        result = await run(ctx, path="a.txt", offset=1, limit=4)
        assert "offset=5" in result.content

    async def test_reading_to_the_end_does_not_advertise_more(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", "one\ntwo\n")
        result = await run(ctx, path="a.txt")
        assert "more remain" not in result.content

    async def test_offset_past_the_end_is_a_failure_not_an_empty_success(
        self, ctx: ToolContext
    ) -> None:
        make(ctx, "a.txt", "one\ntwo\n")
        result = await run(ctx, path="a.txt", offset=99)
        assert result.failed
        assert "2" in (result.error or "")

    async def test_a_huge_file_is_streamed_not_loaded(self, ctx: ToolContext) -> None:
        """A 40 MB log must be readable by offset without exhausting memory."""
        path = ctx.workspace / "big.log"
        with path.open("wb") as fh:
            for index in range(200_000):
                fh.write(f"{index:06d} filler filler filler filler\n".encode())
        result = await run(ctx, path="big.log", offset=150_000, limit=3)
        assert result.ok is True
        assert "150000|149999 filler" in result.content


class TestGuards:
    async def test_binary_files_are_refused_with_a_reason(self, ctx: ToolContext) -> None:
        path = ctx.workspace / "blob.bin"
        path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rJFIF")
        result = await run(ctx, path="blob.bin")
        assert result.failed
        assert "binary" in (result.error or "")

    async def test_a_long_line_is_truncated_with_a_marker(self, ctx: ToolContext) -> None:
        make(ctx, "long.txt", "x" * 9000)
        result = await run(ctx, path="long.txt")
        assert len(result.content) < 9000
        assert "truncated" in result.content

    async def test_limit_is_capped_by_the_tool(self, ctx: ToolContext) -> None:
        make(ctx, "a.txt", "one\n")
        result = await run(ctx, path="a.txt", limit=10**6)
        assert result.failed
        assert "at most" in (result.error or "")

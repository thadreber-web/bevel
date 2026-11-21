"""Tests for the render layer.

The render layer is what the TUI shows the user, and its contract is that it
never fails quietly: a diff it cannot parse, a lexer it cannot find, and a body
it had to shorten all say so in the output. These tests assert that wording,
because "diff unavailable" versus an empty box is the difference between a UI
that tells the truth and one that lies.
"""

from __future__ import annotations

import difflib

from rich.text import Text

from bevel.render.diff import (
    ADD,
    CONTEXT,
    DELETE,
    HEADER,
    HUNK,
    RAW,
    parse_diff,
    render_diff,
)
from bevel.render.markdown import render_markdown, strip_markdown
from bevel.render.syntax import guess_language, highlight_code


def _unified(before: list[str], after: list[str], rel: str = "pkg/mod.py") -> str:
    return "\n".join(difflib.unified_diff(before, after, fromfile=rel, tofile=rel, lineterm=""))


class TestParseDiff:
    def test_write_tool_header_becomes_the_summary_and_path(self) -> None:
        text = "pkg/mod.py: 2 lines changed\n" + _unified(["a", "b"], ["a", "c"])
        parsed = parse_diff(text)
        assert parsed.error is None
        assert parsed.summary == "pkg/mod.py: 2 lines changed"
        assert parsed.path == "pkg/mod.py"
        assert (parsed.added, parsed.removed) == (1, 1)

    def test_singular_wording_in_the_header_is_preserved_verbatim(self) -> None:
        """The tool owns the wording; the parser must not re-derive it."""
        parsed = parse_diff("one.py: 1 line changed\n" + _unified(["a"], ["b"], "one.py"))
        assert parsed.summary == "one.py: 1 line changed"

    def test_bare_difflib_output_parses_without_a_summary(self) -> None:
        parsed = parse_diff(_unified(["a", "b"], ["a", "c"]))
        assert parsed.error is None
        assert parsed.summary == ""
        assert parsed.path == "pkg/mod.py"  # taken from the --- / +++ headers
        assert [line.kind for line in parsed.lines].count(HUNK) == 1
        assert [line.kind for line in parsed.lines].count(HEADER) == 2
        assert (parsed.added, parsed.removed) == (1, 1)

    def test_line_numbers_track_both_sides_from_the_hunk_header(self) -> None:
        before = [f"line{i}" for i in range(1, 11)]
        after = [f"line{i}" if i != 5 else "changed" for i in range(1, 11)]
        parsed = parse_diff(_unified(before, after))
        assert parsed.error is None
        dels = [line for line in parsed.lines if line.kind is DELETE]
        adds = [line for line in parsed.lines if line.kind is ADD]
        assert [(d.old_line, d.new_line) for d in dels] == [(5, None)]
        assert [(a.old_line, a.new_line) for a in adds] == [(None, 5)]
        contexts = [line for line in parsed.lines if line.kind is CONTEXT]
        # Context lines before the change advance both counters together.
        assert [(c.old_line, c.new_line) for c in contexts[:3]] == [(2, 2), (3, 3), (4, 4)]
        # The delete/add pair nets to zero, so the trailing context resumes at
        # the same number on both sides rather than drifting.
        assert [(c.old_line, c.new_line) for c in contexts[3:]] == [(6, 6), (7, 7), (8, 8)]

    def test_hunk_numbers_come_from_the_header_not_from_a_counter(self) -> None:
        parsed = parse_diff("@@ -100,3 +200,3 @@\n context\n-removed\n+added\n")
        assert parsed.error is None
        dels = [line for line in parsed.lines if line.kind is DELETE]
        adds = [line for line in parsed.lines if line.kind is ADD]
        assert dels[0].old_line == 101
        assert adds[0].new_line == 201

    def test_malformed_hunk_header_sets_error_and_names_the_line(self) -> None:
        parsed = parse_diff("@@ -not,a +number,b @@\n context\n")
        assert parsed.error is not None
        assert "malformed hunk header" in parsed.error
        assert "@@ -not,a +number,b @@" in parsed.error
        assert render_diff(parsed).plain.startswith("diff unavailable: malformed hunk header")

    def test_text_with_no_hunks_is_not_a_diff(self) -> None:
        parsed = parse_diff("x.py: 1 line changed\nprose, not a diff\n")
        assert parsed.error == "no hunk headers found; this is not a unified diff"
        assert not parsed.ok
        # Junk between the summary and the first hunk stays visible, not dropped.
        assert [line.kind for line in parsed.lines] == [RAW]

    def test_empty_input_reports_empty_rather_than_rendering_blank(self) -> None:
        for text in ("", "   ", "\n\n"):
            parsed = parse_diff(text)
            assert parsed.error == "no diff text was provided"
            assert render_diff(parsed).plain == "diff unavailable: no diff text was provided"


class TestRenderDiff:
    def test_elision_states_how_much_was_hidden_and_what_was_shown(self) -> None:
        body = "\n".join(["@@ -1,12 +1,12 @@", *[" context" for _ in range(12)]])
        rendered = render_diff(parse_diff(body), max_lines=5)
        assert "[... 8 of 13 diff lines elided (showing the first 5) ...]" in rendered.plain
        # The cap counts every diff line, the hunk header included.
        assert rendered.plain.count("context") == 4

    def test_no_elision_note_when_the_body_fits(self) -> None:
        rendered = render_diff(parse_diff(_unified(["a", "b"], ["a", "c"])), max_lines=50)
        assert "elided" not in rendered.plain

    def test_counts_and_heading_are_rendered(self) -> None:
        text = "x.py: 1 line changed\n" + _unified(["a"], ["b"], "x.py")
        plain = render_diff(parse_diff(text)).plain
        assert "x.py: 1 line changed" in plain
        assert "+1 -1" in plain

    def test_line_numbers_appear_in_the_gutter(self) -> None:
        plain = render_diff(parse_diff(_unified(["a", "b"], ["a", "c"]))).plain
        assert "2   -b" in plain
        assert "  2 +c" in plain

    def test_line_numbers_can_be_turned_off(self) -> None:
        plain = render_diff(
            parse_diff(_unified(["a", "b"], ["a", "c"])), show_line_numbers=False
        ).plain
        assert "-b" in plain
        assert "+c" in plain
        assert "2   -b" not in plain


class TestHighlightCode:
    def test_unknown_language_falls_back_to_plain_text_and_says_what_it_tried(self) -> None:
        text = highlight_code("hello", "klingon")
        assert "hello" in text.plain
        assert "no pygments lexer for 'klingon'; showing plain text" in text.plain

    def test_unknown_filename_falls_back_and_names_the_filename(self) -> None:
        text = highlight_code("hello", filename="notes.zzz")
        assert "no pygments lexer for 'notes.zzz'; showing plain text" in text.plain

    def test_known_language_highlights_without_a_note(self) -> None:
        text = highlight_code("def f():\n    return 1", "python")
        assert "def f():" in text.plain
        assert "plain text" not in text.plain

    def test_elision_is_stated(self) -> None:
        text = highlight_code("\n".join(str(i) for i in range(10)), "python", max_lines=4)
        assert "[... 6 of 10 lines elided (showing the first 4) ...]" in text.plain

    def test_empty_code_renders_empty(self) -> None:
        assert highlight_code("").plain == ""


class TestGuessLanguage:
    def test_pyi_resolves_to_python(self) -> None:
        assert guess_language("stubs/thing.pyi") == "python"

    def test_toml_resolves_to_toml(self) -> None:
        assert guess_language("pyproject.toml") == "toml"

    def test_env_resolves_to_ini(self) -> None:
        assert guess_language(".env") == "ini"

    def test_suffix_overrides_win_case_insensitively(self) -> None:
        assert guess_language("CONFIG.TOML") == "toml"

    def test_unknown_extension_returns_none(self) -> None:
        assert guess_language("notes.zzz") is None

    def test_plain_source_files_use_pygments_lookup(self) -> None:
        assert guess_language("main.py") == "Python"


class TestMarkdown:
    def test_empty_renders_as_empty_text(self) -> None:
        rendered = render_markdown("")
        assert isinstance(rendered, Text)
        assert rendered.plain == ""

    def test_strip_removes_markers(self) -> None:
        assert strip_markdown("# Title\n\n**bold** and `code`") == "Title\nbold and code"

    def test_strip_keeps_link_targets_inline(self) -> None:
        assert strip_markdown("[bevel](https://x.dev)") == "bevel (https://x.dev)"

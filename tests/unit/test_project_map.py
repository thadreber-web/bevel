"""The project map: what the model is told about the workspace before it acts.

The map is the orient step's entire output, so these assert both halves of the
promise in ``docs/agent-loop.md``: it answers the layout questions up front, and
it stays small enough to be worth sending. A map that grows with the repo is the
failure mode -- it would eat the context it exists to save.
"""

from __future__ import annotations

from pathlib import Path

from bevel.core.context.project_map import (
    MAX_ENTRIES_PER_DIR,
    MAX_MAP_CHARS,
    MAX_TREE_LINES,
    build_project_map,
)


def test_map_names_the_root_and_the_git_state(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    text = build_project_map(tmp_path)
    assert str(tmp_path) in text
    assert "not a git repository" in text
    assert "main.py" in text


def test_missing_workspace_is_reported_not_raised(tmp_path: Path) -> None:
    text = build_project_map(tmp_path / "nope")
    assert "does not exist" in text


def test_empty_workspace_says_so(tmp_path: Path) -> None:
    assert "(empty workspace)" in build_project_map(tmp_path)


def test_tree_is_depth_capped(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "too_deep.txt").write_text("x", encoding="utf-8")
    text = build_project_map(tmp_path)
    assert "a/" in text
    assert "b/" in text
    assert "c/" in text
    assert "too_deep.txt" not in text


def test_gitignored_names_are_pruned(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build/\nsecret.txt\n# comment\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.o").write_text("x", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("x", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    text = build_project_map(tmp_path)
    assert "keep.txt" in text
    assert "build" not in text
    assert "secret.txt" not in text


def test_junk_directories_are_pruned(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "src").mkdir()
    text = build_project_map(tmp_path)
    assert "src/" in text
    assert ".venv" not in text
    assert "node_modules" not in text


def test_manifest_head_and_readme_head_are_included(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n' + "".join(f"#{n}\n" for n in range(200)),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# Demo\n\nA thing.\n" + "".join(f"filler {n}\n" for n in range(200)),
        encoding="utf-8",
    )
    text = build_project_map(tmp_path)
    assert "## pyproject.toml" in text
    assert 'name = "demo"' in text
    assert "## README (README.md" in text
    assert "A thing." in text
    assert "continues - read it for the rest" in text
    assert "filler 199" not in text


def test_declared_entry_points_are_parsed(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n\n[project.scripts]\nbevel = "bevel.cli:main"\n',
        encoding="utf-8",
    )
    (tmp_path / "package.json").write_text(
        '{"main": "lib/index.js", "bin": {"demo": "lib/cli.js"}, "scripts": {"test": "jest"}}',
        encoding="utf-8",
    )
    text = build_project_map(tmp_path)
    assert "[project.scripts] bevel = bevel.cli:main" in text
    assert "package.json main: lib/index.js" in text
    assert "package.json bin: demo -> lib/cli.js" in text
    assert "package.json scripts: test" in text


def test_unparseable_manifest_is_announced_not_silently_dropped(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("this is not = = toml\n", encoding="utf-8")
    text = build_project_map(tmp_path)
    assert "not valid TOML" in text


def test_binary_manifest_is_not_dumped(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_bytes(b"\x00\x01\x02\x03\xff")
    text = build_project_map(tmp_path)
    assert "## Makefile" not in text


def test_map_is_bounded_on_a_large_workspace(tmp_path: Path) -> None:
    for n in range(MAX_ENTRIES_PER_DIR + 60):
        (tmp_path / f"file_{n:04d}.py").write_text("x", encoding="utf-8")
    for n in range(MAX_TREE_LINES + 40):
        (tmp_path / f"pkg_{n:03d}").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    text = build_project_map(tmp_path)
    assert len(text) <= MAX_MAP_CHARS + len(
        f"\n[project map cut at {MAX_MAP_CHARS} characters - use glob and read for the rest]"
    )
    assert "more entries" in text

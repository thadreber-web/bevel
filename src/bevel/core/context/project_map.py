"""Cheap repo orientation: the project map the loop hands the model before iteration one.

A model that has never seen the workspace spends its first three calls on ``glob``,
``read`` and ``shell`` rediscovering what every run discovers the same way: the
layout, the manifest, what the README says the thing is. The map answers those up
front for the price of one bounded directory walk.

It is an orientation, not an index. It is built once per run and never refreshed,
it reads the *heads* of manifests and the README rather than all of them, and every
section is capped -- a project map big enough to matter is a project map big enough
to eat the context window it was meant to save. Anything cut says it was cut and
names the tool to use instead.

Gitignore handling is deliberately partial: the workspace's own ``.gitignore`` is
honoured for names, but nested ``.gitignore`` files and negation (``!``) patterns
are not, and a leading ``/`` is treated as unanchored. Pruning here is a
convenience, never a jail -- anything the map hides is one ``glob`` call away.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Final

from bevel.core.tools._fs import SKIP_DIRS, glob_matches, is_binary, relative_to_root

__all__ = ["build_project_map"]

#: Directory levels the tree descends into. Deeper structure is the ``glob`` tool's job.
MAX_DEPTH: Final[int] = 3
#: Entries rendered per directory before the rest are counted rather than listed.
MAX_ENTRIES_PER_DIR: Final[int] = 40
#: Hard cap on tree lines, whatever the per-directory cap allowed.
MAX_TREE_LINES: Final[int] = 120
#: How many manifests get a head, and how many lines each head keeps.
MAX_MANIFESTS: Final[int] = 6
MANIFEST_HEAD_LINES: Final[int] = 40
README_HEAD_LINES: Final[int] = 30
#: Characters kept per line. A minified ``package.json`` is a single very long line.
MAX_LINE_CHARS: Final[int] = 200
#: Whole-map cap. The map exists to save context, not to spend it.
MAX_MAP_CHARS: Final[int] = 6000
#: Bytes sniffed for a NUL before a file is refused as binary.
_SNIFF_BYTES: Final[int] = 4096
#: Bytes read before parsing a structured file. No manifest needs more to be recognised.
_PARSE_CAP_BYTES: Final[int] = 65_536

_MANIFEST_NAMES: Final[tuple[str, ...]] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "CMakeLists.txt",
    "build.gradle",
    "pom.xml",
)

_README_NAMES: Final[tuple[str, ...]] = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "readme.md",
)

#: Root-level files that conventionally start a program.
_ENTRY_FILE_NAMES: Final[tuple[str, ...]] = (
    "main.py",
    "manage.py",
    "app.py",
    "asgi.py",
    "wsgi.py",
    "src/main.py",
    "index.js",
    "index.ts",
    "src/index.ts",
    "cmd/main.go",
)


def build_project_map(workspace: Path) -> str:
    """Everything the model should know about a workspace before it acts on it.

    Returns text and never raises. A workspace that cannot be read is reported *in*
    the map: an agent with no orientation still has to answer a greeting, and a
    crash here would take the run down before the model had said anything.
    """
    root = Path(workspace)
    if not root.is_dir():
        return (
            f"# project map: {root}\nthe workspace does not exist; there is nothing to orient on."
        )

    kind = "git repository" if (root / ".git").exists() else "not a git repository"
    sections = [f"# project map: {root} ({kind})"]

    rules = _gitignore_rules(root)
    tree = _tree(root, rules)
    body = "\n".join(tree) if tree else "  (empty workspace)"
    sections.append(f"## tree (depth {MAX_DEPTH})\n{body}")

    manifests = _manifest_section(root)
    if manifests:
        sections.append("\n".join(manifests))
    readme = _readme_section(root)
    if readme:
        sections.append("\n".join(readme))
    entries = _entry_points(root)
    if entries:
        sections.append("## entry points\n" + "\n".join(entries))

    text = "\n\n".join(sections)
    if len(text) > MAX_MAP_CHARS:
        return (
            text[:MAX_MAP_CHARS]
            + f"\n[project map cut at {MAX_MAP_CHARS} characters - use glob and read for the rest]"
        )
    return text


# --------------------------------------------------------------------------- tree


def _tree(root: Path, rules: tuple[tuple[str, bool], ...]) -> list[str]:
    lines: list[str] = []
    _walk(root, root, "", 1, rules=rules, lines=lines)
    if len(lines) >= MAX_TREE_LINES:
        lines.append(f"  + tree cut at {MAX_TREE_LINES} lines - use glob to see the rest")
    return lines


def _walk(
    root: Path,
    directory: Path,
    prefix: str,
    depth: int,
    *,
    rules: tuple[tuple[str, bool], ...],
    lines: list[str],
) -> None:
    try:
        entries = sorted(directory.iterdir(), key=_sort_key)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        lines.append(f"{prefix}[cannot list {directory.name}: {reason}]")
        return

    visible = [entry for entry in entries if not _ignored(entry, root, rules)]
    for entry in visible[:MAX_ENTRIES_PER_DIR]:
        if len(lines) >= MAX_TREE_LINES:
            return
        if entry.is_dir():
            lines.append(f"{prefix}{entry.name}/")
            if depth < MAX_DEPTH:
                _walk(root, entry, f"{prefix}  ", depth + 1, rules=rules, lines=lines)
        else:
            lines.append(f"{prefix}{entry.name}")
    if len(visible) > MAX_ENTRIES_PER_DIR:
        lines.append(f"{prefix}+ {len(visible) - MAX_ENTRIES_PER_DIR} more entries")


def _sort_key(path: Path) -> tuple[bool, str]:
    """Directories first, then files, both case-insensitively.

    Sorted so that two runs over the same tree produce byte-identical text; a map
    whose order shifts between runs makes the model's citations lie.
    """
    return path.is_file(), path.name.lower()


def _ignored(entry: Path, root: Path, rules: tuple[tuple[str, bool], ...]) -> bool:
    """Is this entry junk, or ignored by the workspace's own ``.gitignore``?

    The junk set is the tools' own (:data:`~bevel.core.tools._fs.SKIP_DIRS`) rather
    than a second list, so the tree and ``glob`` never disagree about what exists.
    """
    name = entry.name
    if name in SKIP_DIRS or name.endswith(".egg-info"):
        return True
    is_dir = entry.is_dir()
    rel = relative_to_root(entry, root)
    for pattern, dir_only in rules:
        if dir_only and not is_dir:
            continue
        subject = rel if "/" in pattern else name
        if glob_matches(pattern, subject):
            return True
    return False


def _gitignore_rules(root: Path) -> tuple[tuple[str, bool], ...]:
    """Patterns from the workspace's own ``.gitignore``, as ``(pattern, dir_only)``.

    Comments, blanks and negations are dropped and a leading ``/`` is stripped, so
    ``/build/`` behaves like ``build`` and matches at any depth. Nested gitignore
    files are not read: the map is a glance, not a git reimplementation.
    """
    text = _read_text(root / ".gitignore")
    if text is None:
        return ()
    rules: list[tuple[str, bool]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        dir_only = line.endswith("/")
        pattern = line.strip("/").strip()
        if pattern:
            rules.append((pattern, dir_only))
    return tuple(rules)


# ------------------------------------------------------------------------ files


def _read_text(path: Path, cap_bytes: int = _PARSE_CAP_BYTES) -> str | None:
    """Small text file as a string, or ``None`` if it is absent or binary."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(cap_bytes)
    except OSError:
        return None
    if is_binary(raw):
        return None
    return raw.decode("utf-8", errors="replace")


def _head(path: Path, max_lines: int) -> tuple[list[str], bool]:
    """First ``max_lines`` lines of ``path``, and whether more remain.

    Streams rather than slurps: the README of a real project can be enormous and the
    map only ever wants its opening. Returns ``([], False)`` for a file that is
    absent or binary, which the caller reads as "no section".
    """
    try:
        with path.open("rb") as handle:
            if is_binary(handle.read(_SNIFF_BYTES)):
                return [], False
        kept: list[str] = []
        more = False
        # utf-8-sig strips a BOM and decodes plain UTF-8 otherwise; one bad byte
        # should cost a character, not the section.
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                if len(kept) >= max_lines:
                    more = True
                    break
                text = line.rstrip("\r\n")
                kept.append(text if len(text) <= MAX_LINE_CHARS else text[:MAX_LINE_CHARS] + " ...")
        return kept, more
    except OSError:
        return [], False


def _manifest_section(root: Path) -> list[str]:
    lines: list[str] = []
    shown = 0
    for name in _MANIFEST_NAMES:
        if shown >= MAX_MANIFESTS:
            break
        body, more = _head(root / name, MANIFEST_HEAD_LINES)
        if not body:
            continue
        shown += 1
        lines.append(f"## {name}")
        lines.extend(f"  {line}" for line in body)
        if more:
            lines.append(f"  ... [{name} continues - read it for the rest]")
    return lines


def _readme_section(root: Path) -> list[str]:
    for name in _README_NAMES:
        body, more = _head(root / name, README_HEAD_LINES)
        if not body:
            continue
        lines = [f"## README ({name}, first {README_HEAD_LINES} lines)"]
        lines.extend(f"  {line}" for line in body)
        if more:
            lines.append(f"  ... [{name} continues - read it for the rest]")
        return lines
    return []


def _load(path: Path, loader: Callable[[str], object], label: str, lines: list[str]) -> object:
    """Parse a structured file, reporting a bad one into the map instead of raising.

    A manifest that will not parse is still information: a silently absent section
    reads to the model as "this project declares nothing".
    """
    text = _read_text(path)
    if text is None:
        return None
    try:
        return loader(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        lines.append(f"  [{path.name} is not valid {label}: {exc}]")
        return None


def _string_map(value: object) -> dict[str, str]:
    """A mapping of strings to strings; anything else counts as absent."""
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)}


def _entry_points(root: Path) -> list[str]:
    """Declared and conventional ways into the program.

    Both structured files are parsed rather than grepped, because a script named
    ``test`` and a script whose *value* contains the word ``test`` are not the same
    thing.
    """
    lines: list[str] = []

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = _load(pyproject, tomllib.loads, "TOML", lines)
        project = data.get("project") if isinstance(data, dict) else None
        scripts = project.get("scripts") if isinstance(project, dict) else None
        for name, target in sorted(_string_map(scripts).items()):
            lines.append(f"  [project.scripts] {name} = {target}")

    package = root / "package.json"
    if package.is_file():
        payload = _load(package, json.loads, "JSON", lines)
        if isinstance(payload, dict):
            main = payload.get("main")
            if isinstance(main, str):
                lines.append(f"  package.json main: {main}")
            bin_field = payload.get("bin")
            if isinstance(bin_field, str):
                lines.append(f"  package.json bin: {bin_field}")
            for name, target in sorted(_string_map(bin_field).items()):
                lines.append(f"  package.json bin: {name} -> {target}")
            names = sorted(_string_map(payload.get("scripts")))
            if names:
                lines.append("  package.json scripts: " + ", ".join(names))

    conventional = [name for name in _ENTRY_FILE_NAMES if (root / name).is_file()]
    if conventional:
        lines.append("  files: " + ", ".join(conventional))
    return lines

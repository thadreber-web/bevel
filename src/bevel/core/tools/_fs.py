"""Filesystem details shared by ``read``, ``grep`` and ``glob``.

Private on purpose. These are the three decisions every filesystem tool would
otherwise make slightly differently on its own: which directories are junk,
what counts as a binary file, and how a user-facing glob pattern turns into a
match test. Divergence there is the kind of bug users report as "grep found
something glob did not".
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

#: Directories that are never interesting to a coding agent and are large
#: enough that walking them costs real time. Deliberately excludes ``build``
#: and ``dist``: some repositories keep source there, and a search that
#: silently hides files is worse than one that is a little slow.
SKIP_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".eggs",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
    }
)

#: Bytes sniffed for a NUL when deciding whether a file is text.
_SNIFF_BYTES: Final[int] = 8192


def is_binary(sample: bytes) -> bool:
    """Is this the start of a binary file?

    A NUL byte is the test every mainstream tool uses. Encoding-detection
    libraries are not available under the three-dependency rule, and guessing
    at text encodings to produce mojibake is worse than refusing.
    """
    return b"\x00" in sample


def iter_files(root: Path, *, max_files: int = 20_000) -> Iterator[Path]:
    """Yield files under ``root`` in a stable order, pruning junk directories.

    ``os.walk`` rather than :meth:`pathlib.Path.glob` because only ``walk``
    lets us prune ``node_modules`` *before* descending into it. Sorted so that
    two runs over the same tree produce byte-identical output -- a tool whose
    result order shifts between calls makes a model's line references lie.
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if not _is_junk(name))
        base = Path(dirpath)
        for name in sorted(filenames):
            if _is_junk(name):
                continue
            if count >= max_files:
                return
            count += 1
            yield base / name


def _is_junk(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def glob_matches(pattern: str, rel_path: str) -> bool:
    """Match a root-relative posix path against a glob pattern.

    Two rules, both copied from ``rg --glob`` / ``gitignore`` because that is
    what users expect: a pattern with no separator matches the basename, so
    ``*.py`` finds ``src/app.py``; and a pattern with a separator is anchored
    at the *end* of the path rather than the start, so ``util/*.py`` finds
    ``src/util/helper.py`` while ``src/*.py`` still refuses to reach into
    ``src/util/``.
    """
    if "/" not in pattern:
        subject = rel_path.rsplit("/", 1)[-1]
        return _glob_regex(pattern).match(subject) is not None
    return _glob_regex(pattern).match(rel_path) is not None


def relative_to_root(path: Path, root: Path) -> str:
    """``path`` relative to ``root``, as posix text, for glob matching.

    ``os.path.relpath`` rather than :meth:`Path.relative_to` because a symlinked
    or case-mismatched path must degrade to a relative string, not raise.
    """
    return Path(os.path.relpath(path, root)).as_posix()


_GLOB_CACHE: dict[str, re.Pattern[str]] = {}


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob to a regex.

    Written by hand instead of using :func:`fnmatch.translate`, which lets
    ``*`` cross path separators -- ``src/*.py`` would then match
    ``src/pkg/app.py`` and quietly over-report. Cached because grep applies the
    same pattern to every file in the tree.
    """
    cached = _GLOB_CACHE.get(pattern)
    if cached is not None:
        return cached
    parts: list[str] = []
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "*":
            if pattern[index : index + 2] == "**":
                parts.append(".*")
                index += 2
                # ``a/**/b`` must also match ``a/b``, so swallow the separator.
                if pattern[index : index + 1] == "/":
                    index += 1
            else:
                parts.append("[^/]*")
                index += 1
        elif char == "?":
            parts.append("[^/]")
            index += 1
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end < 0:
                parts.append(re.escape(char))
                index += 1
            else:
                parts.append(pattern[index : end + 1].replace("\\", "\\\\"))
                index = end + 1
        else:
            parts.append(re.escape(char))
            index += 1
    # A separator in the pattern means "anywhere below", not "from the root":
    # gitignore and ``rg --glob`` both let ``util/*.py`` find
    # ``src/util/helper.py``, and users expect the same here.
    anchor = r"(?:.*/)?" if "/" in pattern else ""
    compiled = re.compile("".join((r"^", anchor, *parts, r"$")))
    _GLOB_CACHE[pattern] = compiled
    return compiled

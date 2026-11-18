"""The rules the engine decides with: built-in patterns and learned allowlists.

Two questions are answered here and nowhere else. *Is this command dangerous?*
is answered by a built-in table. *Has the user already allowed this command?*
is answered by ``policy.db``, which is where "always allow ``git status``" from
an approval prompt has to survive to the next session or it is a lie.

Matching is deliberately literal. A rule says "this argv starts with these
tokens" and either it does or it does not; there is no shell parsing, no
globbing, and no attempt to understand what a command means. A pattern the
table misses leaves the call at ``EXEC_UNSAFE``, which still asks — the cost of
an incomplete table is one extra prompt, never a silent pass.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "LearnedRule",
    "RuleStore",
    "match_destructive",
    "match_safe",
    "normalise_argv",
    "rule_hash",
]

_WINDOWS = os.name == "nt"
_EXECUTABLE_SUFFIXES = (".exe", ".cmd", ".bat", ".com")


def normalise_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Canonical form of a command for rule matching.

    Drops empty arguments, and on Windows lowercases the program and drops its
    executable suffix, so ``Git.exe STATUS`` and ``git status`` are the same
    command to a rule. Arguments are left alone: ``--format={hash}`` means what
    it says, and folding its case would change the command rather than
    canonicalise it.
    """
    items = [item for item in argv if item]
    if not items:
        return ()
    program = items[0].strip().lower()
    if _WINDOWS:
        for suffix in _EXECUTABLE_SUFFIXES:
            if program.endswith(suffix):
                program = program[: -len(suffix)]
                break
    return (program, *items[1:])


@dataclass(frozen=True)
class _Rule:
    """One entry of a built-in table.

    ``prefix`` is matched token-by-token against the normalised argv; every
    ``fragments`` entry must appear in the lowercased joined command, except a
    two-character flag like ``-r`` which a bundle of short flags also satisfies.
    A rule with both must satisfy both.
    """

    name: str
    prefix: tuple[str, ...] = ()
    fragments: tuple[str, ...] = ()


def _matches(rule: _Rule, argv: tuple[str, ...]) -> bool:
    if not rule.prefix and not rule.fragments:
        return False
    if rule.prefix:
        if len(argv) < len(rule.prefix):
            return False
        if tuple(token.lower() for token in argv[: len(rule.prefix)]) != rule.prefix:
            return False
    joined = " ".join(argv).lower()
    for fragment in rule.fragments:
        if fragment in joined:
            continue
        # ``rm -rf`` asks for both ``-r`` and ``-f``, and neither is a substring of
        # the joined command. A single-letter flag may therefore also be satisfied
        # by a bundle of short flags. Longer fragments (``of=/dev/``, ``| sh``)
        # still have to appear verbatim, which is what lets them span tokens.
        if (
            len(fragment) == 2
            and fragment.startswith("-")
            and any(
                token.startswith("-")
                and not token.startswith("--")
                and fragment[1] in token[1:].lower()
                for token in argv
            )
        ):
            continue
        return False
    return True


def _match(table: tuple[_Rule, ...], argv: Sequence[str]) -> str | None:
    normalised = normalise_argv(argv)
    if not normalised:
        return None
    for rule in table:
        if _matches(rule, normalised):
            return rule.name
    return None


# Commands that only look, so they can run without asking. The bar is "reading
# cannot change anything": nothing here builds, installs, or runs repository
# code, which is why ``pytest`` is absent even though it feels harmless — it
# executes whatever is in the repo.
_SAFE_RULES: tuple[_Rule, ...] = (
    _Rule("git status", ("git", "status")),
    _Rule("git diff", ("git", "diff")),
    _Rule("git log", ("git", "log")),
    _Rule("git show", ("git", "show")),
    _Rule("git blame", ("git", "blame")),
    _Rule("git branch", ("git", "branch")),
    _Rule("git rev-parse", ("git", "rev-parse")),
    _Rule("git ls-files", ("git", "ls-files")),
    _Rule("git remote", ("git", "remote")),
    _Rule("ls", ("ls",)),
    _Rule("dir", ("dir",)),
    _Rule("cat", ("cat",)),
    _Rule("type", ("type",)),
    _Rule("head", ("head",)),
    _Rule("tail", ("tail",)),
    _Rule("wc", ("wc",)),
    _Rule("tree", ("tree",)),
    _Rule("find", ("find",)),
    _Rule("rg", ("rg",)),
    _Rule("grep", ("grep",)),
    _Rule("pwd", ("pwd",)),
    _Rule("echo", ("echo",)),
    _Rule("which", ("which",)),
    _Rule("where", ("where",)),
    _Rule("whoami", ("whoami",)),
    _Rule("uname", ("uname",)),
    _Rule("hostname", ("hostname",)),
    _Rule("date", ("date",)),
    _Rule("python --version", ("python", "--version")),
    _Rule("python3 --version", ("python3", "--version")),
    _Rule("pip show", ("pip", "show")),
    _Rule("uv --version", ("uv", "--version")),
    _Rule("node --version", ("node", "--version")),
)

# Commands that destroy, or that hand a shell to whatever a download returns.
# Fragments are matched on the joined command because the damage usually lives
# in the flags, and every fragment here is lowercase to match the join.
_DESTRUCTIVE_RULES: tuple[_Rule, ...] = (
    _Rule("rm recursive force", ("rm",), ("-r", "-f")),
    _Rule("rm --no-preserve-root", ("rm",), ("--no-preserve-root",)),
    _Rule("del /s", ("del",), ("/s",)),
    _Rule("Remove-Item -Recurse -Force", (), ("remove-item", "-recurse", "-force")),
    _Rule("Format-Volume", (), ("format-volume",)),
    _Rule("git push --force", ("git", "push"), ("-f",)),
    _Rule("git reset --hard", ("git", "reset"), ("--hard",)),
    _Rule("git clean -f", ("git", "clean"), ("-f",)),
    _Rule("git checkout discard", ("git", "checkout", ".")),
    _Rule("git restore .", ("git", "restore", ".")),
    _Rule("git branch -D", ("git", "branch", "-d")),
    _Rule("git gc --prune", ("git", "gc"), ("--prune=now",)),
    _Rule("mkfs", ("mkfs",)),
    _Rule("fdisk", ("fdisk",)),
    _Rule("diskpart", ("diskpart",)),
    _Rule("format", ("format",)),
    _Rule("dd to device", ("dd",), ("of=/dev/",)),
    _Rule("chmod -R 777", ("chmod",), ("-r", "777")),
    _Rule("chown -R", ("chown",), ("-r", "/")),
    _Rule("reg delete", ("reg", "delete")),
    _Rule("vssadmin delete shadows", ("vssadmin", "delete", "shadows")),
    _Rule("schtasks delete", ("schtasks",), ("/delete",)),
    _Rule("net stop", ("net", "stop")),
    _Rule("shutdown", ("shutdown",)),
    # Fragments are ANDed, so this asks for a pipe and a shell rather than listing
    # spellings of a pipe. ``sh`` is inside ``bash`` and ``zsh``, and matching the
    # join covers ``| sh`` and ``|sh`` alike.
    _Rule("curl piped to shell", ("curl",), ("|", "sh")),
    _Rule("wget piped to shell", ("wget",), ("|", "sh")),
    _Rule("fork bomb", (), (":|:&",)),
)


def match_safe(argv: Sequence[str]) -> str | None:
    """Name of the built-in read-only rule this argv satisfies, if any."""
    return _match(_SAFE_RULES, argv)


def match_destructive(argv: Sequence[str]) -> str | None:
    """Name of the built-in destructive rule this argv satisfies, if any."""
    return _match(_DESTRUCTIVE_RULES, argv)


@dataclass(frozen=True)
class LearnedRule:
    """One entry the user granted with "always allow".

    ``hash`` is the rule's identity — stable under the case and separator noise
    that ``argv`` carries — and ``created_at`` is what makes a years-old grant
    visible as such when someone reviews the list.
    """

    argv: tuple[str, ...]
    hash: str
    created_at: str


def rule_hash(argv: Sequence[str]) -> str:
    """Stable identity of a command: SHA-256 over the normalised argv."""
    return hashlib.sha256("\0".join(normalise_argv(argv)).encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS learned_rules (
    hash       TEXT PRIMARY KEY,
    argv       TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


class RuleStore:
    """``policy.db``: the commands the user said to stop asking about.

    Prefix matching, so "always allow ``git``" covers ``git status`` and
    ``git diff`` alike. That breadth is what the phrase means to a user, and it
    is bounded by the engine checking destructive patterns *first* — a learned
    rule can make a command free, never make a destructive one free.

    The connection opens on first use, so constructing an agent does not create
    a database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._db_path)
            self._connection.execute(_SCHEMA)
            self._connection.commit()
        return self._connection

    def lookup(self, argv: Sequence[str]) -> LearnedRule | None:
        """The learned rule that covers ``argv``, longest prefix first.

        Longest first so a specific grant (``git push --dry-run``) is reported
        rather than a broader one (``git``) that also happens to cover it.
        """
        normalised = normalise_argv(argv)
        if not normalised:
            return None
        rows = (
            self._connect().execute("SELECT hash, argv, created_at FROM learned_rules").fetchall()
        )
        best: LearnedRule | None = None
        for row in rows:
            rule = _row_to_rule(row)
            if len(rule.argv) > len(normalised):
                continue
            if tuple(token.lower() for token in normalised[: len(rule.argv)]) != rule.argv:
                continue
            if best is None or len(rule.argv) > len(best.argv):
                best = rule
        return best

    def remember(self, argv: Sequence[str]) -> LearnedRule:
        """Persist ``argv`` as allowed. Re-granting a command changes nothing."""
        normalised = normalise_argv(argv)
        if not normalised:
            raise ValueError("cannot learn an empty command")
        hash_value = rule_hash(normalised)
        connection = self._connect()
        existing = connection.execute(
            "SELECT created_at FROM learned_rules WHERE hash = ?", (hash_value,)
        ).fetchone()
        if existing is not None:
            return LearnedRule(normalised, hash_value, str(existing[0]))
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        connection.execute(
            "INSERT INTO learned_rules (hash, argv, created_at) VALUES (?, ?, ?)",
            (hash_value, shlex.join(normalised), created_at),
        )
        connection.commit()
        return LearnedRule(normalised, hash_value, created_at)

    def revoke(self, argv: Sequence[str]) -> bool:
        """Forget an exact learned rule. True if there was something to forget."""
        normalised = normalise_argv(argv)
        if not normalised:
            return False
        connection = self._connect()
        cursor = connection.execute(
            "DELETE FROM learned_rules WHERE hash = ?", (rule_hash(normalised),)
        )
        connection.commit()
        return cursor.rowcount > 0

    def entries(self) -> tuple[LearnedRule, ...]:
        """Every learned rule, oldest first — what ``bevel policy list`` shows."""
        rows = (
            self._connect()
            .execute("SELECT hash, argv, created_at FROM learned_rules ORDER BY created_at, hash")
            .fetchall()
        )
        return tuple(_row_to_rule(row) for row in rows)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def _row_to_rule(row: tuple[str, str, str]) -> LearnedRule:
    hash_value, stored, created_at = row
    return LearnedRule(normalise_argv(shlex.split(stored)), hash_value, created_at)

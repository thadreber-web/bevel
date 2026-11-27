"""The ``git`` tool: a narrow interpreter, not a shell.

Does bevel need a ``git`` tool when the model can already run git through
``shell`` and the policy engine already frees ``git status``? Yes, and only for
one reason: ``shell`` runs *any program*, and a tool is narrower than that in
exactly the three ways that matter here.

1. **The cwd is pinned to the workspace.** There is no ``cwd`` parameter and no
   way to set one, so the path jail in :class:`~bevel.core.tools.base.ToolContext`
   is the only directory git can ever be pointed at.
2. **The subcommands are an allowlist.** Anything not named in
   :data:`ALLOWED_SUBCOMMANDS` is refused before it reaches git.
3. **The argv that escapes the workspace or executes arbitrary code is refused.**
   git is not a read-only tool; it is an interpreter with a large argv surface,
   and several of its options are load-bearing parts of that surface. So the
   refusals below are an **allowlist of what is permitted**, expressed as the
   small set of options that survive — never a denylist of known-bad tokens,
   because a denylist is one un-listed spelling away from a bypass.

The refused options are refused *before* any child is spawned, and the child is
never handed a model-controlled environment: :func:`bevel.core._git.run_git`
gives it the parent environment plus nothing else.

Why the tier is ``EXEC_UNSAFE`` and the argv is exposed as ``command``
---------------------------------------------------------------------
This is the part that is not guessable. :meth:`PolicyEngine._draft` computes an
argv *only* for a tool whose tier is in ``_EXEC_TIERS``, reading it from
``arguments["command"]``. Declaring ``EXEC_UNSAFE`` and naming the argv
``command`` is what buys, for free: the destructive rules for ``git push
--force`` / ``git reset --hard`` / ``git clean -f`` / ``git checkout .`` /
``git restore .`` / ``git branch -D`` / ``git gc --prune``, the free rules for
``git status`` / ``diff`` / ``log`` / ``show`` / ``blame`` / ``branch`` /
``rev-parse`` / ``ls-files`` / ``remote``, and an ask for everything else. A
``git`` tool that declared ``READ`` would get none of that — it would skip the
argv matching entirely and hand the model ``git reset --hard`` for free.

The four decisions, written down
-------------------------------
**Trailer.** PLAN says "commit with trailer" without saying whose or when. The
trailer is ``Assisted-by: bevel``, appended as the final paragraph of every
commit this tool makes, and only when it is not already present — so a retried
commit does not stack them. It is appended, never substituted: a message the
user dictated survives byte-for-byte above it. The trailer is metadata about
provenance; rewriting the author's words would be the lie.

**Identity.** :mod:`bevel.core.checkpoint.snapshot` forges
``GIT_AUTHOR_*``/``GIT_COMMITTER_*`` so a *checkpoint* never depends on the
machine. This tool does the opposite on purpose: a commit in the user's own
history authored by ``bevel <bevel@localhost>`` would be a false statement about
who wrote it, so the identity is left entirely to the user's git config. The
cost of that choice is that a machine with no ``user.name`` would otherwise fail
with git's own opaque error, so the tool pre-checks the identity and returns a
:class:`ToolResult.failure` naming the exact fix (``git config user.name ...``)
rather than a non-zero exit the model has to interpret.

**Branch hygiene.** The tool reports the branch a commit landed on (git's own
commit output names it), and refuses ``git switch -c`` / ``git branch <name>``
when that name already exists — creating over an existing branch is how a model
silently destroys a line of work. It does **not** validate a branch name beyond
what git itself rejects, does **not** auto-create a branch, and does **not**
offer force-push: force-push is m11b (PR operations), and its absence is stated
here rather than left silent.

**Shared runner.** This tool does not spawn git itself; it calls
:func:`bevel.core._git.run_git`, the single git spawner extracted from the two
private ones that existed before (``checkpoint/snapshot.py`` and
``memory/store.py``). Three copies of a tree-kill-and-timeout dance was the
wrong number; one is right, and the reason is in that module.

Assumptions pinned by tests (stated rather than asked)
------------------------------------------------------
- ``add`` must name explicit paths (``--`` then paths); a bare ``git add .`` is
  allowed but the model is nudged toward paths by the schema description.
- ``git push`` is **not** on the allowlist in m11a at all — pushing is m11b.
- ``commit`` requires ``-m``; there is no editor and no ``-F``/``--file``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from bevel.core._git import run_git
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec

#: The provenance line appended to every commit this tool makes. Lowercased
#: comparison in :meth:`GitTool._with_trailer` is what stops it doubling.
TRAILER: Final[str] = "Assisted-by: bevel"

#: Subcommands the tool will run at all. Everything else is refused. Reads are
#: the git subcommands that only look; mutating ones change the working tree or
#: the index and so are the ones that trigger a checkpoint (see ``mutates``).
READ_SUBCOMMANDS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "diff",
        "log",
        "show",
        "blame",
        "branch",
        "rev-parse",
        "rev-list",
        "ls-files",
        "ls-tree",
        "remote",
        "describe",
        "shortlog",
        "cat-file",
    }
)
MUTATING_SUBCOMMANDS: Final[frozenset[str]] = frozenset(
    {
        "add",
        "commit",
        "checkout",
        "switch",
        "restore",
        "reset",
        "stash",
        "mv",
        "rm",
        "tag",
    }
)
ALLOWED_SUBCOMMANDS: Final[frozenset[str]] = READ_SUBCOMMANDS | MUTATING_SUBCOMMANDS

#: Options refused wherever they appear. Each is a way for argv to either
#: execute a program, write a file, or bypass the user's own guardrails — none
#: is a read, and none is needed by the allowlisted subcommands.
#:
#: - ``--ext-diff`` / ``--textconv`` run a configured external filter (code exec)
#: - ``--output`` writes a diff to a file (a write outside the jail's view)
#: - ``--upload-pack`` / ``--receive-pack`` name a program to run for a transfer
#: - ``--no-verify`` skips the user's hooks — the user's guardrails are the user's
#: - ``-F`` / ``--file`` read a commit message from a file (arbitrary file read)
#: - ``-C`` / ``--git-dir`` / ``--work-tree`` / ``--namespace`` relocate the repo,
#:   which defeats the path jail — the only thing confining the model today
#: - ``--exec-path`` redirects git to a different set of helper programs
_REFUSED_ANYWHERE: Final[tuple[str, ...]] = (
    "--ext-diff",
    "--textconv",
    "--output",
    "--upload-pack",
    "--receive-pack",
    "--no-verify",
    "-F",
    "--file",
    "-C",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
)

#: A URL scheme that makes git run an arbitrary program to move bytes.
_EXT_TRANSPORT: Final[str] = "ext::"

#: Global options that are harmless enough to permit before the subcommand.
_ALLOWED_GLOBALS: Final[frozenset[str]] = frozenset({"--no-pager"})


class GitTool:
    """Run an allowlisted ``git`` subcommand in the workspace."""

    spec: Final[ToolSpec] = ToolSpec(
        name="git",
        description=(
            "Run a git subcommand in the workspace. `command` is an argv array whose "
            "first element is 'git' and second is the subcommand (status, diff, log, "
            "show, add, commit, branch, switch, ...). There is no cwd: git always runs "
            "at the workspace root. Repo-relocating options (-C, --git-dir, "
            "--work-tree), config injection (-c), and options that execute programs "
            "or write files (--ext-diff, --output, --no-verify, -F) are refused. "
            "commit requires -m; a trailer is appended and the author is taken from "
            "your git config. push, fetch, pull and clone are not available."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "description": (
                        "argv beginning with 'git' then a subcommand, e.g. "
                        '["git", "commit", "-m", "message"]. Name the paths to add '
                        'explicitly (["git", "add", "--", "file.py"]) rather than ".".'
                    ),
                    "minItems": 2,
                    "items": {"type": "string"},
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.EXEC_UNSAFE
    is_parallel_safe: Final[bool] = False

    def mutates(self, arguments: Mapping[str, Any]) -> bool:
        """Whether this call changes the working tree, so the loop checkpoints first.

        Read from the subcommand before execution: a ``commit`` that later fails
        still cost a checkpoint, which is the conservative direction — an undone
        mutation is cheaper than an un-undoable one.
        """
        subcommand = _subcommand_of(_argv_of(arguments))
        return subcommand in MUTATING_SUBCOMMANDS

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        argv = _argv_of(arguments)
        subcommand, refusal = _validate(argv)
        if subcommand is None:
            return ToolResult.failure(refusal or "git: refused")

        if subcommand in MUTATING_SUBCOMMANDS:
            not_root = await _refuse_if_not_repo_root(ctx)
            if not_root is not None:
                return ToolResult.failure(not_root)

        if subcommand == "commit":
            return await self._commit(argv, ctx)

        collision = await _branch_collision(argv, subcommand, ctx)
        if collision is not None:
            return ToolResult.failure(collision)

        result = await run_git(ctx.workspace, argv[1:])
        return ToolResult(ok=True, content=_render(result.out, result.err, result.code))

    async def _commit(self, argv: list[str], ctx: ToolContext) -> ToolResult:
        if not _has_message(argv):
            return ToolResult.failure(
                "git commit needs a message: pass -m with the message as its own "
                "argv element (there is no editor, and -F/--file are refused)."
            )
        detached = await _refuse_if_detached(ctx)
        if detached is not None:
            return ToolResult.failure(detached)
        missing = await _missing_identity(ctx)
        if missing is not None:
            return ToolResult.failure(missing)

        message = _message_from(argv)
        full = _with_trailer(message)
        # Rebuild argv as `git commit -m <full>` so the model cannot smuggle a
        # second -m or an option we have not reasoned about past the message.
        result = await run_git(ctx.workspace, ["commit", "-m", full])
        return ToolResult(ok=True, content=_render(result.out, result.err, result.code))


# ------------------------------------------------------------ branch hygiene


def _new_branch_name(argv: list[str], subcommand: str) -> str | None:
    """The branch a call would *create*, or ``None`` if it creates none.

    ``git switch -c <name>``, ``git checkout -b <name>`` and ``git branch
    <name>`` are the three creation shapes. ``git branch`` with no argument
    lists, and ``git branch -d`` deletes (the latter is the engine's destructive
    rule, not a creation). One return per shape is the point; collapsing them
    would make the shapes harder to read, not easier.
    """
    if subcommand == "switch":
        return _value_after(argv, ("-c", "--create"))
    if subcommand == "checkout":
        return _value_after(argv, ("-b", "-B", "--branch"))
    if subcommand == "branch":
        rest = argv[2:]
        # A bare ``git branch <name>`` creates. Anything with a flag -- --list,
        # -d, -m, -a -- is a list, delete, or move, not a creation, and is left
        # to git (and, for the destructive spellings, to the policy engine).
        if any(arg.startswith("-") for arg in rest):
            return None
        return rest[0] if rest else None
    return None


def _value_after(argv: list[str], flags: tuple[str, ...]) -> str | None:
    """The argument following one of ``flags``, or ``None`` if it is absent."""
    for index, arg in enumerate(argv):
        if arg in flags and index + 1 < len(argv):
            return argv[index + 1]
    return None


async def _branch_collision(argv: list[str], subcommand: str, ctx: ToolContext) -> str | None:
    """Refuse to create a branch whose name is already taken.

    Creating over an existing branch is how a model silently destroys a line of
    work, and git's own behaviour differs between the two shapes (``branch
    <existing>`` errors, ``switch -c <existing>`` errors too — but the model
    learns that only after a spawn). Checking first names the branch and why.
    """
    name = _new_branch_name(argv, subcommand)
    if name is None:
        return None
    result = await run_git(
        ctx.workspace, ["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"]
    )
    if result.code == 0 and result.out.strip():
        return (
            f"git: branch {name!r} already exists, and creating over it would "
            f"discard that work. Switch to it with ['git', 'switch', {name!r}] "
            "instead, or pick another name."
        )
    return None


# ------------------------------------------------------------------ validation


def _argv_of(arguments: Mapping[str, Any]) -> list[str]:
    raw = arguments.get("command")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def _subcommand_of(argv: list[str]) -> str | None:
    """The first non-option token after the program name, or ``None``.

    Global options precede the subcommand in git's grammar, so the subcommand is
    the first token that is not an option.
    """
    for token in argv[1:]:
        if not token.startswith("-"):
            return token
    return None


def _validate(argv: list[str]) -> tuple[str | None, str]:  # noqa: PLR0911
    """The subcommand that may run, or ``None`` plus the reason it may not.

    Runs entirely in-process: nothing here spawns, so a refusal costs no child.
    One return per distinct refusal, each naming its own reason — collapsing
    them (the way ``snapshot()`` refuses to) would make the failures vaguer.
    """
    if not argv:
        return None, "git: 'command' must be an argv array beginning with 'git'"
    program = argv[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    if program.removesuffix(".exe") != "git":
        return None, (
            f"git: the first element must be 'git', not {argv[0]!r}. This tool runs "
            "git only; use the shell tool for any other program."
        )

    subcommand = _subcommand_of(argv)
    if subcommand is None:
        return None, "git: no subcommand given (the argv must name one after 'git')"
    if subcommand not in ALLOWED_SUBCOMMANDS:
        return None, (
            f"git: subcommand {subcommand!r} is not allowed. Allowed: "
            f"{', '.join(sorted(ALLOWED_SUBCOMMANDS))}. push/fetch/pull/clone are "
            "not available in this milestone."
        )

    for index, token in enumerate(argv[1:], start=1):
        if _EXT_TRANSPORT in token:
            return None, (
                f"git: refusing the {_EXT_TRANSPORT!r} transport ({token!r}) — it runs "
                "an arbitrary program to move bytes."
            )
        for refused in _REFUSED_ANYWHERE:
            if token == refused or token.startswith(f"{refused}="):
                return None, _refusal_message(refused)
        if _is_config_option(token) and (index < argv.index(subcommand) or subcommand != "switch"):
            # ``-c key=value`` injects config, and core.pager/editor/fsmonitor/
            # sshCommand/gitProxy/askPass/credential.helper all run programs. It
            # is refused in global position; ``switch -c`` (create branch) is the
            # one legitimate use after the subcommand.
            return None, _refusal_message("-c")
        if index < argv.index(subcommand) and token not in _ALLOWED_GLOBALS:
            # Any other option before the subcommand is a global we have not
            # reasoned about. Refuse it rather than guess.
            return None, (
                f"git: the global option {token!r} is not allowed. Global options "
                "that relocate the repo or change git's behaviour are refused."
            )
    return subcommand, ""


def _is_config_option(arg: str) -> bool:
    """``-c`` or a bundled ``-c<key>=<value>``, but not ``--check`` and friends."""
    return arg == "-c" or (arg.startswith("-c") and not arg.startswith("--"))


def _refusal_message(token: str) -> str:
    why = {
        "-C": "it relocates the repository, defeating the workspace path jail",
        "--git-dir": "it relocates the repository, defeating the workspace path jail",
        "--work-tree": "it relocates the working tree, defeating the workspace path jail",
        "--namespace": "it changes which refs git sees, defeating the workspace path jail",
        "-c": "it injects git config, and several config keys run arbitrary programs",
        "--exec-path": "it redirects git to a different set of helper programs",
        "--upload-pack": "it names a program for git to run during a transfer",
        "--receive-pack": "it names a program for git to run during a transfer",
        "--output": "it writes a file outside the workspace's view",
        "--ext-diff": "it runs a configured external filter program",
        "--textconv": "it runs a configured external filter program",
        "--no-verify": "it skips your hooks, which are your guardrails",
        "-F": "it reads a commit message from a file; use -m",
        "--file": "it reads a commit message from a file; use -m",
    }.get(token, "it is not on the allowlist")
    return f"git: refusing {token!r} — {why}."


# --------------------------------------------------------------------- commit


def _has_message(argv: list[str]) -> bool:
    return any(
        arg in ("-m", "--message") or (arg.startswith("-m") and not arg.startswith("--"))
        for arg in argv
    )


def _message_from(argv: list[str]) -> str:
    """The message the model passed to ``-m``/``--message`` (last one wins)."""
    message = ""
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("-m", "--message") and index + 1 < len(argv):
            message = argv[index + 1]
            index += 2
            continue
        if arg.startswith("-m") and not arg.startswith("--") and len(arg) > 2:
            message = arg[2:]
        index += 1
    return message


def _with_trailer(message: str) -> str:
    """Append the provenance trailer unless it is already there.

    The user's message is never edited — the trailer is added below it, after a
    blank line, which is git's own trailer convention.
    """
    if TRAILER.lower() in message.lower():
        return message
    stripped = message.rstrip("\n")
    return f"{stripped}\n\n{TRAILER}"


async def _missing_identity(ctx: ToolContext) -> str | None:
    """Explain the missing git identity, or ``None`` if the user has one.

    The tool deliberately does not forge an identity (see the module docstring),
    so a machine with no ``user.name`` has to be told what to set rather than
    handed git's opaque exit code.
    """
    missing: list[str] = []
    for key, fix in (
        ("user.name", 'git config user.name "Your Name"'),
        ("user.email", "git config user.email you@example.com"),
    ):
        result = await run_git(ctx.workspace, ["config", "--get", key])
        if result.code != 0 or not result.out.strip():
            missing.append(f"{key} (run: {fix})")
    if not missing:
        return None
    return (
        "git commit refused: your repository has no commit identity, so a commit "
        "here would be misattributed. Set " + "; ".join(missing) + " and retry."
    )


# ------------------------------------------------------------------- repo root


async def _refuse_if_detached(ctx: ToolContext) -> str | None:
    """Refuse a commit on a detached HEAD.

    A commit made while detached lands on no branch, which is work the user
    cannot find later. ``symbolic-ref`` is git's own answer to "is HEAD a
    branch", so there is no ref-name guessing here.
    """
    result = await run_git(ctx.workspace, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    if result.code != 0:
        return (
            "git commit refused: HEAD is detached, so this commit would land on no "
            "branch and be hard to find later. Create or switch to a branch first "
            "(['git', 'switch', '-c', '<name>'])."
        )
    return None


async def _refuse_if_not_repo_root(ctx: ToolContext) -> str | None:
    """Refuse a mutating call made from a subdirectory of the repo.

    A mutating git command runs against the whole repository regardless of the
    cwd it was started in, so a workspace that is a subdirectory would let a
    model mutate files it never handed bevel. ``--show-prefix`` is git's own
    verdict on whether the cwd is the root.
    """
    result = await run_git(ctx.workspace, ["rev-parse", "--show-prefix"])
    if result.code != 0:
        reason = result.err.strip().splitlines()[0] if result.err.strip() else "no output"
        return f"git: {ctx.workspace} is not a git repository ({reason})"
    if result.out.strip():
        return (
            f"git: {ctx.workspace} is a subdirectory of the repository, and this "
            "subcommand would change the whole repo. Run bevel at the repository root."
        )
    return None


# --------------------------------------------------------------------- render


def _render(out: str, err: str, code: int) -> str:
    parts: list[str] = []
    if out.strip():
        parts.append(out.rstrip("\n"))
    if err.strip():
        parts.append(f"[stderr]\n{err.rstrip(chr(10))}")
    parts.append(f"[exit {code}]")
    return "\n".join(parts)


__all__ = [
    "ALLOWED_SUBCOMMANDS",
    "MUTATING_SUBCOMMANDS",
    "READ_SUBCOMMANDS",
    "TRAILER",
    "GitTool",
]

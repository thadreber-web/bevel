"""Session checkpoints: a snapshot of a dirty tree, and a way back to it.

This is the whole of milestone m06. It is deliberately small, because almost
all of it is work ``git`` already does. The part that is not is *how* the
snapshot is taken.

The problem bevel has: the user's work-in-progress is uncommitted, so HEAD is
not a safe place to roll back to. ``git stash`` moves the user's own work
around, a branch reset throws it away, and ``git worktree add`` cannot carry
uncommitted changes into the new tree at all.

The trick, and the only real logic here: build the commit from a throwaway
index (``GIT_INDEX_FILE`` pointed at a temporary file *outside* the repository)
and point a side ref at it. The user's index, branch, HEAD and stash are never
touched, nothing is checked out, and the working tree is not modified.

``restore`` is the one operation that does touch the tree. Verified against
real git on Windows: ``git restore --source=<commit> -- .`` rewrites modified
files, recreates files the agent deleted, and recreates the user's untracked
work-in-progress, all without staging anything or moving HEAD.

There is no database. The ref and its commit message are the metadata.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from bevel.core._process import TreeKiller, close_process, uses_new_session

__all__ = [
    "CheckpointResult",
    "RestoreResult",
    "checkpoint_ref",
    "list_checkpoints",
    "restore",
    "snapshot",
]

_GIT_TIMEOUT_SECONDS: Final[float] = 30.0
_DRAIN_GRACE_SECONDS: Final[float] = 2.0

#: Checkpoints live under this namespace so they cannot collide with a user's
#: branch names, and ``git for-each-ref 'refs/heads/bevel/checkpoint/*'`` lists
#: every one of them.
_REF_PREFIX: Final[str] = "refs/heads/bevel/checkpoint/"

#: Committer identity for snapshot commits, set explicitly so that a checkpoint
#: does not depend on the machine having a configured ``user.name``.
_IDENTITY: Final[dict[str, str]] = {
    "GIT_AUTHOR_NAME": "bevel",
    "GIT_AUTHOR_EMAIL": "bevel@localhost",
    "GIT_COMMITTER_NAME": "bevel",
    "GIT_COMMITTER_EMAIL": "bevel@localhost",
}

_SESSION_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class CheckpointResult:
    """Outcome of a snapshot attempt.

    ``ok`` is True only when a checkpoint ref exists afterwards — newly created
    or already present from an earlier run. Anything else is a loud failure: a
    checkpoint the user believes they can undo but cannot is worse than none.
    """

    ok: bool
    commit: str | None = None
    error: str | None = None
    summary: str = ""
    already_present: bool = False


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of a restore.

    ``removed`` lists files this call deleted — the one part of a restore the
    user cannot undo, so it is always reported. ``untracked`` lists untracked
    files the agent created that were left in place; they are deleted only when
    the caller explicitly asks, because bevel cannot tell a file the agent wrote
    from one the user wrote while bevel was thinking.
    """

    ok: bool
    commit: str | None = None
    error: str | None = None
    untracked: tuple[str, ...] = field(default=())
    removed: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class _GitResult:
    code: int
    out: str
    err: str


def checkpoint_ref(session: str) -> str:
    """Return the ref a session's checkpoint lives on."""
    cleaned = _SESSION_CHARS.sub("-", session).strip("-.") or "session"
    return f"{_REF_PREFIX}{cleaned}"


# --------------------------------------------------------------------------- git


async def _git(
    workspace: Path,
    args: Sequence[str],
    extra_env: dict[str, str] | None = None,
) -> _GitResult:
    """Run one git command. Never raises; the caller reads ``code``.

    A missing or broken git comes back as a non-zero result instead of an
    exception, so a machine without git degrades to "no checkpoint" rather than
    crashing the agent.
    """
    argv = ["git", *args]
    env = {**os.environ, **(extra_env or {})}
    proc: asyncio.subprocess.Process | None = None
    try:
        kwargs: dict[str, Any] = {
            "cwd": str(workspace),
            "stdin": asyncio.subprocess.DEVNULL,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
            "env": env,
        }
        if uses_new_session():
            kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(*argv, **kwargs)
    except (OSError, ValueError, RuntimeError) as exc:
        return _GitResult(127, "", f"cannot run git: {exc}")

    killer = TreeKiller(proc.pid)
    try:
        try:
            raw_out, raw_err = await asyncio.wait_for(
                proc.communicate(), timeout=_GIT_TIMEOUT_SECONDS
            )
        except TimeoutError:
            killer.kill()
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(proc.communicate(), timeout=_DRAIN_GRACE_SECONDS)
            return _GitResult(124, "", f"git {' '.join(args)} timed out")
        return _GitResult(
            proc.returncode if proc.returncode is not None else 124,
            raw_out.decode("utf-8", "replace"),
            raw_err.decode("utf-8", "replace"),
        )
    except asyncio.CancelledError:
        killer.kill()
        raise
    finally:
        killer.close()
        close_process(proc)


def _reason(res: _GitResult) -> str:
    stripped = res.err.strip()
    return stripped.splitlines()[0] if stripped else "no output"


def _paths(text: str) -> tuple[str, ...]:
    """Split NUL-separated git output into paths."""
    return tuple(part for part in text.split("\0") if part)


async def _refusal(workspace: Path) -> str | None:
    """Explain why ``workspace`` cannot be checkpointed, or ``None`` if it can.

    Checkpointing a subdirectory would snapshot the whole repository while
    looking like it snapshotted the workspace, and restore would rewrite files
    the user never handed bevel. Refusing is the honest answer. ``--show-prefix``
    is git's own verdict on whether the cwd is the root, so it beats comparing
    paths: no case or symlink hazards on Windows, and nothing blocking on the
    event loop. A workspace that does not exist fails here too, when git cannot
    be started in it.
    """
    res = await _git(workspace, ["rev-parse", "--show-toplevel", "--show-prefix"])
    if res.code != 0:
        return _reason(res)
    toplevel, _, prefix = res.out.partition("\n")
    if not toplevel.strip():
        return "not a git repository"
    if prefix.strip():
        return f"{workspace} is not the repository root ({toplevel.strip()})"
    return None


# ---------------------------------------------------------------------- snapshot


async def snapshot(workspace: Path, session: str) -> CheckpointResult:  # noqa: PLR0911
    """Commit the working tree as it stands, WIP included, onto the session's ref.

    The user's index, branch and stash are untouched: the commit is built from a
    temporary index and only the side ref moves. Idempotent — if the session
    already has a checkpoint it is reported back rather than overwritten, so the
    snapshot always describes the tree as it was when the session started.
    """
    # Each return below is a different fail-closed reason; collapsing them to
    # satisfy pylint's return-count heuristic would make the failures vaguer.
    reason = await _refusal(workspace)
    if reason:
        return CheckpointResult(ok=False, error=f"no checkpoint taken: {reason}")

    ref = checkpoint_ref(session)
    existing = await _git(workspace, ["rev-parse", "--verify", "--quiet", ref])
    if existing.code == 0 and existing.out.strip():
        sha = existing.out.strip()
        return CheckpointResult(
            ok=True,
            commit=sha,
            summary=f"checkpoint already present for {session} at {sha[:8]}",
            already_present=True,
        )

    parent = (await _git(workspace, ["rev-parse", "--verify", "--quiet", "HEAD"])).out.strip()
    if not parent:
        return CheckpointResult(
            ok=False,
            error="no checkpoint taken: the repository has no commits, so there is nothing to "
            "snapshot against",
        )

    # The index has to live outside the repository: inside it, `git add --all`
    # sweeps the index file itself into the snapshot.
    with tempfile.TemporaryDirectory(prefix="bevel-checkpoint-") as tmp:
        env = {**_IDENTITY, "GIT_INDEX_FILE": str(Path(tmp) / "index")}

        add = await _git(workspace, ["add", "--all", "--", "."], extra_env=env)
        if add.code != 0:
            return CheckpointResult(
                ok=False, error=f"no checkpoint taken: git add failed: {_reason(add)}"
            )

        write = await _git(workspace, ["write-tree"], extra_env=env)
        if write.code != 0 or not write.out.strip():
            return CheckpointResult(
                ok=False, error=f"no checkpoint taken: git write-tree failed: {_reason(write)}"
            )

        commit = await _git(
            workspace,
            [
                "commit-tree",
                write.out.strip(),
                "-p",
                parent,
                "-m",
                f"bevel checkpoint: {session}",
            ],
            extra_env=env,
        )
        if commit.code != 0 or not commit.out.strip():
            return CheckpointResult(
                ok=False, error=f"no checkpoint taken: git commit-tree failed: {_reason(commit)}"
            )
        sha = commit.out.strip()

        update = await _git(workspace, ["update-ref", ref, sha])
        if update.code != 0:
            return CheckpointResult(
                ok=False, error=f"no checkpoint taken: git update-ref failed: {_reason(update)}"
            )
        return CheckpointResult(ok=True, commit=sha, summary=f"checkpointed {session} at {sha[:8]}")


# ----------------------------------------------------------------------- restore


async def list_checkpoints(workspace: Path) -> tuple[tuple[str, str], ...]:
    """Return ``(session, commit)`` for every checkpoint in ``workspace``, newest first."""
    res = await _git(
        workspace,
        [
            "for-each-ref",
            "--format=%(committerdate)T\t%(refname)\t%(objectname)",
            "--sort=-committerdate",
            _REF_PREFIX,
        ],
    )
    if res.code != 0:
        return ()
    found: list[tuple[str, str]] = []
    for line in res.out.splitlines():
        fields = line.strip().split("\t")
        if len(fields) != 3:
            continue
        found.append((fields[1][len(_REF_PREFIX) :], fields[2]))
    return tuple(found)


async def _snapshot_paths(workspace: Path, commit: str) -> tuple[frozenset[str], str]:
    """Paths recorded in a snapshot commit, or an error string."""
    res = await _git(workspace, ["ls-tree", "-r", "--name-only", "-z", commit])
    if res.code != 0:
        return frozenset(), _reason(res)
    return frozenset(_paths(res.out)), ""


async def _untracked(workspace: Path) -> tuple[tuple[str, ...], str]:
    """Untracked, non-ignored paths, or an error string."""
    res = await _git(workspace, ["ls-files", "--others", "--exclude-standard", "-z"])
    if res.code != 0:
        return (), _reason(res)
    return _paths(res.out), ""


async def restore(  # noqa: PLR0911
    workspace: Path,
    session: str | None = None,
    remove_untracked: bool = False,
) -> RestoreResult:
    """Rewind the working tree to a session's checkpoint.

    ``session`` selects the checkpoint; the most recent one is used when it is
    omitted. Tracked content goes back to the snapshot, and so does the user's
    untracked work-in-progress, which git recreates for free because it is in
    the snapshot tree.

    Files created after the snapshot are left alone unless ``remove_untracked``
    is set — bevel cannot tell a file the agent wrote from one the user wrote
    while bevel was thinking. Anything deleted is listed in ``removed``. Nothing
    is staged and HEAD does not move.
    """
    reason = await _refusal(workspace)
    if reason:
        return RestoreResult(ok=False, error=f"nothing restored: {reason}")

    commit: str
    if session is None:
        recent = await list_checkpoints(workspace)
        if not recent:
            return RestoreResult(
                ok=False, error="nothing restored: this workspace has no checkpoints"
            )
        commit = recent[0][1]
    else:
        resolved = await _git(
            workspace, ["rev-parse", "--verify", "--quiet", checkpoint_ref(session)]
        )
        if resolved.code != 0 or not resolved.out.strip():
            return RestoreResult(ok=False, error=f"nothing restored: no checkpoint for {session!r}")
        commit = resolved.out.strip()

    # Work out what is safe to delete *before* touching anything, and refuse to
    # delete a single file unless both listings succeeded.
    in_snapshot, err = await _snapshot_paths(workspace, commit)
    if err:
        return RestoreResult(ok=False, commit=commit, error=f"nothing restored: {err}")
    untracked, err = await _untracked(workspace)
    if err:
        return RestoreResult(ok=False, commit=commit, error=f"nothing restored: {err}")
    new_since_snapshot = tuple(path for path in untracked if path not in in_snapshot)

    done = await _git(workspace, ["restore", f"--source={commit}", "--", "."])
    if done.code != 0:
        return RestoreResult(
            ok=False, commit=commit, error=f"nothing restored: git restore failed: {_reason(done)}"
        )

    removed: tuple[str, ...] = ()
    if remove_untracked and new_since_snapshot:
        cleaned = await _git(workspace, ["clean", "-f", "--", *new_since_snapshot])
        if cleaned.code != 0:
            return RestoreResult(
                ok=False,
                commit=commit,
                untracked=new_since_snapshot,
                error=f"tracked files restored but git clean failed: {_reason(cleaned)}",
            )
        removed = new_since_snapshot

    return RestoreResult(ok=True, commit=commit, untracked=new_since_snapshot, removed=removed)

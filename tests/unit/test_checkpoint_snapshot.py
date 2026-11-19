"""Tests for the m06 checkpoint snapshot and restore.

Everything here drives a **real** git repository in ``tmp_path``. Mocking git
would prove only that the code agrees with our own guess about git, and the
whole milestone rests on git semantics that were measured rather than assumed:
a temporary index outside the repo, ``commit-tree`` onto a side ref, and
``restore --source=`` for the way back.

The properties that matter most, and so are asserted most directly:

* the snapshot contains the user's uncommitted work, and *not* gitignored files,
* taking one leaves the user's index and HEAD exactly where they were,
* restore brings back modified files, deleted files, and the user's untracked
  work-in-progress, while leaving files the agent invented alone unless asked,
* every failure is loud — a failed result always carries a reason.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from bevel.core.checkpoint import (
    CheckpointResult,
    RestoreResult,
    checkpoint_ref,
    list_checkpoints,
    restore,
    snapshot,
)

_IDENTITY_ARGS = (
    "-c",
    "user.name=bevel",
    "-c",
    "user.email=bevel@localhost",
    "-c",
    "commit.gpgsign=false",
)


def write(repo: Path, name: str, text: str) -> None:
    """Write a file with no newline translation.

    ``Path.write_text`` rewrites ``\\n`` as ``\\r\\n`` on Windows, which would make
    the blob bytes disagree with the text the test later compares ``git show``
    against.
    """
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def read(repo: Path, name: str) -> str:
    return (repo / name).read_text(encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    """Run git synchronously in ``repo`` for setup and assertions."""
    proc = subprocess.run(
        ["git", *_IDENTITY_ARGS, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
        check=False,
    )
    if proc.returncode != 0:
        msg = f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        raise AssertionError(msg)
    return proc.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real repository with one commit and a clean tree.

    ``ignored/`` exists so that a snapshot which wrongly passed ``--force`` to
    ``git add`` would sweep build output into the checkpoint and be caught.
    """
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write(root, "tracked.txt", "original\n")
    write(root, "notes.md", "# notes\n")
    write(root, ".gitignore", "ignored/\n")
    write(root, "ignored/cache.bin", "do not snapshot me\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "seed")
    return root


# ------------------------------------------------------------------- ref naming


def test_ref_is_namespaced_under_bevel() -> None:
    assert checkpoint_ref("alpha") == "refs/heads/bevel/checkpoint/alpha"


def test_ref_sanitises_characters_git_would_reject() -> None:
    assert checkpoint_ref("my session ~1").endswith("/my-session-1")
    assert checkpoint_ref("a/b").endswith("/a-b")


def test_ref_never_becomes_empty() -> None:
    assert checkpoint_ref("///").endswith("/session")


# --------------------------------------------------------------------- snapshot


async def test_snapshot_creates_a_commit_on_the_session_ref(repo: Path) -> None:
    res = await snapshot(repo, "alpha")

    assert res.ok, res.error
    assert res.commit
    assert res.summary.startswith("checkpointed alpha at ")
    assert git(repo, "rev-parse", checkpoint_ref("alpha")).strip() == res.commit


async def test_snapshot_records_the_commit_message(repo: Path) -> None:
    res = await snapshot(repo, "alpha")

    assert res.commit
    assert git(repo, "log", "-1", "--format=%s", res.commit).strip() == "bevel checkpoint: alpha"


async def test_snapshot_includes_the_users_uncommitted_work(repo: Path) -> None:
    """The whole point: HEAD is not a safe rollback target, so it is not used."""
    write(repo, "tracked.txt", "user wip\n")
    write(repo, "user-wip.txt", "mine\n")

    res = await snapshot(repo, "alpha")

    assert res.commit
    files = set(git(repo, "ls-tree", "-r", "--name-only", res.commit).splitlines())
    assert {"tracked.txt", "user-wip.txt", "notes.md", ".gitignore"} <= files
    assert git(repo, "show", f"{res.commit}:tracked.txt") == "user wip\n"


async def test_snapshot_leaves_gitignored_files_out(repo: Path) -> None:
    res = await snapshot(repo, "alpha")

    assert res.commit
    files = set(git(repo, "ls-tree", "-r", "--name-only", res.commit).splitlines())
    assert "ignored/cache.bin" not in files


async def test_snapshot_does_not_disturb_the_index_or_head(repo: Path) -> None:
    """A checkpoint must be invisible to the user's own git state.

    ``notes.md`` is deliberately staged first: if the snapshot used the real
    index, that staging would be consumed or rewritten.
    """
    write(repo, "notes.md", "# staged\n")
    git(repo, "add", "notes.md")
    head_before = git(repo, "rev-parse", "HEAD").strip()
    status_before = git(repo, "status", "--porcelain")

    res = await snapshot(repo, "alpha")

    assert res.ok, res.error
    assert git(repo, "rev-parse", "HEAD").strip() == head_before
    assert git(repo, "status", "--porcelain") == status_before
    assert git(repo, "diff", "--cached", "--name-only").split() == ["notes.md"]


async def test_snapshot_is_idempotent_and_keeps_the_original(repo: Path) -> None:
    """A second call must not re-point the ref at a later, less useful tree."""
    first = await snapshot(repo, "alpha")
    write(repo, "tracked.txt", "agent rewrite\n")
    second = await snapshot(repo, "alpha")

    assert first.ok
    assert second.ok
    assert second.already_present
    assert second.commit == first.commit
    assert second.summary.startswith("checkpoint already present for alpha ")
    assert git(repo, "show", f"{first.commit}:tracked.txt") == "original\n"


async def test_snapshot_on_detached_head_still_works(repo: Path) -> None:
    git(repo, "checkout", "-q", "--detach")

    res = await snapshot(repo, "alpha")

    assert res.ok, res.error
    assert res.commit


async def test_snapshot_refuses_a_subdirectory_of_a_repository(repo: Path) -> None:
    """Snapshotting a subdir would rewrite the whole repo on restore."""
    sub = repo / "pkg"
    sub.mkdir()

    res = await snapshot(sub, "alpha")

    assert not res.ok
    assert res.error
    assert "not the repository root" in res.error


async def test_snapshot_without_a_repository_fails_loudly(tmp_path: Path) -> None:
    res = await snapshot(tmp_path, "alpha")

    assert not res.ok
    assert res.commit is None
    assert res.error
    assert res.error.startswith("no checkpoint taken:")


async def test_snapshot_without_any_commit_fails_loudly(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "a.txt", "a\n")

    res = await snapshot(tmp_path, "alpha")

    assert not res.ok
    assert res.error
    assert "no commits" in res.error


async def test_snapshot_degrades_when_git_is_missing(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No git binary means no checkpoint, not a crash in the agent loop."""

    async def no_git(*args: object, **kwargs: object) -> None:
        raise OSError("git is not installed")

    monkeypatch.setattr("asyncio.create_subprocess_exec", no_git)
    res = await snapshot(repo, "alpha")

    assert isinstance(res, CheckpointResult)
    assert not res.ok
    assert res.error
    assert res.error.startswith("no checkpoint taken:")
    assert "git is not installed" in res.error


# ---------------------------------------------------------------------- restore


async def test_restore_round_trip(repo: Path) -> None:
    """Mutate, restore, assert content — the definition of the milestone.

    Covers all four cases at once: a modified tracked file goes back, a deleted
    tracked file comes back, the user's untracked WIP comes back, and a file the
    agent invented stays put.
    """
    write(repo, "tracked.txt", "user wip\n")
    write(repo, "user-wip.txt", "mine\n")
    snap = await snapshot(repo, "alpha")
    assert snap.ok, snap.error

    write(repo, "tracked.txt", "agent rewrite\n")
    (repo / "notes.md").unlink()
    (repo / "user-wip.txt").unlink()
    write(repo, "agent-new.txt", "generated\n")

    res = await restore(repo, "alpha")

    assert res.ok, res.error
    assert res.commit == snap.commit
    assert read(repo, "tracked.txt") == "user wip\n"
    assert read(repo, "notes.md") == "# notes\n"
    assert read(repo, "user-wip.txt") == "mine\n"
    assert (repo / "agent-new.txt").exists()
    assert res.untracked == ("agent-new.txt",)
    assert res.removed == ()


async def test_restore_can_remove_files_the_agent_created(repo: Path) -> None:
    snap = await snapshot(repo, "alpha")
    assert snap.ok, snap.error
    write(repo, "agent-new.txt", "generated\n")

    res = await restore(repo, "alpha", remove_untracked=True)

    assert res.ok, res.error
    assert res.removed == ("agent-new.txt",)
    assert not (repo / "agent-new.txt").exists()


async def test_restore_never_deletes_the_users_own_untracked_files(repo: Path) -> None:
    """The one catastrophic failure mode, tested head-on.

    ``user-wip.txt`` is untracked on disk before and after the snapshot. A
    restore that cleaned "all untracked files" would destroy work nobody asked
    it to touch.
    """
    write(repo, "user-wip.txt", "mine\n")
    snap = await snapshot(repo, "alpha")
    assert snap.ok, snap.error
    write(repo, "agent-new.txt", "generated\n")

    res = await restore(repo, "alpha", remove_untracked=True)

    assert res.ok, res.error
    assert read(repo, "user-wip.txt") == "mine\n"
    assert res.removed == ("agent-new.txt",)


async def test_restore_does_not_stage_anything_or_move_head(repo: Path) -> None:
    head_before = git(repo, "rev-parse", "HEAD").strip()
    snap = await snapshot(repo, "alpha")
    assert snap.ok, snap.error
    write(repo, "tracked.txt", "agent rewrite\n")

    res = await restore(repo, "alpha")

    assert res.ok, res.error
    assert git(repo, "rev-parse", "HEAD").strip() == head_before
    assert git(repo, "diff", "--cached", "--name-only") == ""


async def test_restore_without_a_session_uses_the_newest_checkpoint(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2024-01-01T00:00:00 +0000")
    write(repo, "tracked.txt", "one\n")
    first = await snapshot(repo, "older")
    assert first.ok, first.error
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2024-06-01T00:00:00 +0000")
    write(repo, "tracked.txt", "two\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "--no-verify", "-m", "user commit")
    second = await snapshot(repo, "newer")
    assert second.ok, second.error
    monkeypatch.delenv("GIT_COMMITTER_DATE")
    write(repo, "tracked.txt", "three\n")

    res = await restore(repo)

    assert res.ok, res.error
    assert res.commit == second.commit
    assert read(repo, "tracked.txt") == "two\n"


async def test_restore_reports_an_unknown_session(repo: Path) -> None:
    res = await restore(repo, "never-taken")

    assert not res.ok
    assert res.error
    assert "no checkpoint for" in res.error


async def test_restore_without_any_checkpoint_reports_it(repo: Path) -> None:
    res = await restore(repo)

    assert not res.ok
    assert res.error
    assert "no checkpoints" in res.error


async def test_restore_without_a_repository_fails_loudly(tmp_path: Path) -> None:
    res = await restore(tmp_path, "alpha")

    assert isinstance(res, RestoreResult)
    assert not res.ok
    assert res.error
    assert res.error.startswith("nothing restored:")


# --------------------------------------------------------------- listing refs


async def test_list_checkpoints_returns_session_and_commit_newest_first(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2024-01-01T00:00:00 +0000")
    older = await snapshot(repo, "older")
    monkeypatch.setenv("GIT_COMMITTER_DATE", "2024-06-01T00:00:00 +0000")
    newer = await snapshot(repo, "newer")

    assert older.ok
    assert newer.ok
    assert await list_checkpoints(repo) == (("newer", newer.commit), ("older", older.commit))


async def test_list_checkpoints_is_empty_without_a_repository(tmp_path: Path) -> None:
    assert await list_checkpoints(tmp_path) == ()

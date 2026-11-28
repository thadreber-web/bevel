"""Tests for ``bevel.core.pr`` — the gh-backed PR surface (m11b).

The oracle is ``tests/fixtures/gh.py``: a real Python program spawned as a
real subprocess, exactly the way m10 drives ``tests/fixtures/mcp_server.py``.
Mocking the subprocess layer would prove only that the code agrees with its
own guesses about gh; a real child is the only thing that can prove argv
construction, JSON parsing, exit-code mapping, and the timeout/tree-kill path.

What these tests cannot prove — and the module docstring of ``core/pr.py``
says so plainly — is that GitHub's API accepts what gh sends. That question
belongs to ``tests/integration/test_live_pr.py``, gated on
``BEVEL_GH_INTEGRATION`` the way the live-endpoint tests gate on
``BEVEL_BASE_URL``. Nothing in this file reaches the network: every gh is the
fixture.

The fixture is injected as the ``gh`` program: the module functions take a
``gh`` parameter and the CLI test monkeypatches ``bevel.cli._GH``. bevel's own
validator still checks the *logical* argv, which must begin with the literal
``"gh"``; the spawn replaces the program name, so the fixture sees the argv
after it. The fixture refuses anything off the allowlist with exit 4 rather
than echoing it, so a leak of ``--web`` or ``--repo`` fails loudly on both
sides.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from bevel.core import pr as pr_module
from bevel.core.pr import (
    GH,
    auth_status,
    create_pr,
    update_pr,
    validate_gh_argv,
)
from bevel.core.tools import registry

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gh.py"
FIXTURE_ARGV = [sys.executable, str(FIXTURE)]

#: Generous enough that a slow windows-latest runner never trips them, small
#: enough that a hang fails this file in seconds rather than at pytest's ceiling.
HANG_TIMEOUT = 3.0

_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_IDENTITY_ARGS = (
    "-c",
    "user.name=bevel",
    "-c",
    "user.email=bevel@localhost",
    "-c",
    "commit.gpgsign=false",
)


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


def process_alive(pid: int) -> bool:
    """Best-effort liveness check without adding a dependency (as test_mcp_stdio)."""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def wait_until_dead(pid: int, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"pid {pid} is still alive — the tree kill missed it")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real repository on a pushed ``feature`` branch, with an isolated git config.

    Pushed, because the whole point of the push check is that ``create`` only
    runs when the head branch is on the remote; the refused cases are built on
    top of this fixture. ``remote set-head`` gives the repo a resolvable
    default branch, which is what a real clone has and what the same-branch
    check reads.
    """
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    (root / "tracked.txt").write_text("original\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "seed")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "commit.gpgsign", "false")

    bare = tmp_path / "origin.git"
    bare.mkdir()
    git(bare, "init", "--bare", "-q", "-b", "main")
    git(root, "remote", "add", "origin", str(bare))
    git(root, "push", "-q", "-u", "origin", "main")
    git(root, "remote", "set-head", "origin", "-a")
    git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("work\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "do the thing")
    git(root, "push", "-q", "-u", "origin", "feature")
    return root


@pytest.fixture
def unpushed_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository whose current branch has never been pushed anywhere."""
    root = tmp_path / "lonely"
    root.mkdir()
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git(root, "init", "-q", "-b", "solo")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "seed")
    return root


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where the fixture records what it was asked to do."""
    path = tmp_path / "gh-seen.jsonl"
    monkeypatch.setenv("BEVEL_GH_FIXTURE_SEEN", str(path))
    return path


def rows(seen: Path) -> list[dict[str, Any]]:
    """What the fixture actually received, parsed from its own record file."""
    if not seen.exists():
        return []
    return [json.loads(line) for line in seen.read_text(encoding="utf-8").splitlines() if line]


def mode(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("BEVEL_GH_FIXTURE_MODE", value)


# ------------------------------------------------------------------ declaration


class TestDeclaration:
    def test_the_default_program_is_gh_itself(self) -> None:
        """The plan's chosen backend, by name. No httpx path to GitHub exists."""
        assert GH == ("gh",)

    def test_pr_is_not_a_model_facing_tool(self) -> None:
        """The tool-or-not decision, pinned: PR is a human-only CLI operation.

        If this ever flips, the four pin files (test_tools_registry, tui/test_app,
        test_mcp_bridge, test_mcp_servers, test_mcp_resources) must change in the
        same commit, and the tier question in the module docstring has to be
        answered before the registry accepts it.
        """
        assert registry.get("pr") is None
        assert "pr" not in [tool.spec.name for tool in registry.tools()]


# ------------------------------------------------------------- argv validation


class TestArgvSurface:
    """The allowlist, refused before any child exists.

    ``validate_gh_argv`` is the pure gate ``create_pr``/``update_pr`` run every
    argv through before spawning; each refusal here names its own token.
    """

    def test_a_compliant_create_passes(self) -> None:
        argv = ["gh", "pr", "create", "--title", "t", "--body", "b", "--json", "url"]
        assert validate_gh_argv(argv) == ""

    def test_a_compliant_edit_passes(self) -> None:
        assert validate_gh_argv(["gh", "pr", "edit", "7", "--title", "t", "--ready"]) == ""

    def test_auth_status_passes(self) -> None:
        assert validate_gh_argv(["gh", "auth", "status"]) == ""

    def test_only_gh_may_be_spawned(self) -> None:
        refusal = validate_gh_argv(["curl", "pr", "create"])
        assert "curl" in refusal
        assert "gh" in refusal

    def test_gh_api_is_refused(self) -> None:
        """``gh api`` is raw curl to GitHub — the arbitrary-escape hatch."""
        refusal = validate_gh_argv(["gh", "api", "repos/x/y/pulls", "-X", "POST"])
        assert "api" in refusal

    def test_web_is_refused(self) -> None:
        """``--web`` opens a browser; nothing in bevel may block on a GUI."""
        refusal = validate_gh_argv(["gh", "pr", "create", "--web"])
        assert "--web" in refusal

    def test_repo_is_refused(self) -> None:
        """``--repo`` retargets the repository — the gh analogue of ``git -C``."""
        refusal = validate_gh_argv(["gh", "pr", "create", "--repo", "someone/else"])
        assert "--repo" in refusal

    def test_dry_run_is_refused(self) -> None:
        """A dry run inside bevel would be reported to the user as a real PR.

        The decision is CLI-only: a human who wants a dry run runs gh directly;
        bevel never builds the token, so the gate refuses it outright.
        """
        refusal = validate_gh_argv(["gh", "pr", "create", "--dry-run"])
        assert "--dry-run" in refusal

    @pytest.mark.parametrize(
        "subcommand",
        ["merge", "close", "checkout", "comment", "review", "ready"],
    )
    def test_off_allowlist_pr_subcommands_are_refused(self, subcommand: str) -> None:
        refusal = validate_gh_argv(["gh", "pr", subcommand])
        assert subcommand in refusal

    def test_an_unknown_option_is_refused(self) -> None:
        refusal = validate_gh_argv(["gh", "pr", "create", "--monitor"])
        assert "--monitor" in refusal

    def test_an_unknown_group_is_refused(self) -> None:
        refusal = validate_gh_argv(["gh", "release", "create"])
        assert "release" in refusal

    def test_auth_beyond_status_is_refused(self) -> None:
        """bevel probes auth; it never logs in, logs out, or refreshes tokens."""
        refusal = validate_gh_argv(["gh", "auth", "login"])
        assert "login" in refusal

    def test_edit_needs_a_numeric_number(self) -> None:
        refusal = validate_gh_argv(["gh", "pr", "edit", "7; rm -rf /", "--title", "t"])
        assert "number" in refusal

    def test_the_empty_argv_is_refused(self) -> None:
        assert validate_gh_argv([]) != ""


# ------------------------------------------------------------------ create: argv


class TestCreateArgv:
    async def test_a_bare_create_uses_fill_and_the_json_projection(
        self, repo: Path, seen: Path
    ) -> None:
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        (auth_row, create_row) = rows(seen)
        assert auth_row["argv"] == ["auth", "status"]
        assert create_row["argv"] == [
            "pr",
            "create",
            "--head",
            "feature",
            "--fill",
            "--json",
            "url,number,title,state",
        ]

    async def test_the_body_comes_from_the_commits_and_bevel_never_writes_one(
        self, repo: Path, seen: Path
    ) -> None:
        """The trailer decision, pinned: bevel authors no PR body.

        ``--fill`` lets gh compose the body from the commit messages, so the
        ``Assisted-by: bevel`` line m11a put in the commit flows into the PR
        unchanged — provenance through the commit, never through a bevel-
        injected PR line. bevel never sends ``--body`` and never appends the
        trailer itself.
        """
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        argv = rows(seen)[-1]["argv"]
        assert "--fill" in argv
        assert "--body" not in argv
        assert "Assisted-by" not in json.dumps(argv)

    async def test_a_user_dictated_body_survives_unchanged(self, repo: Path, seen: Path) -> None:
        """The other half of the trailer decision: what the user wrote is verbatim.

        bevel appends nothing to a body it was handed -- no trailer, no footer,
        no reflow. The commit trailer is the provenance channel; the PR body is
        the author's words.
        """
        body = "Because:\n- reasons\n\nSigned off by whoever."
        result = await create_pr(repo, title="Add the thing", body=body, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        argv = rows(seen)[-1]["argv"]
        assert argv[4:8] == ["--title", "Add the thing", "--body", body]
        assert "--fill" not in argv, "--fill would overwrite a body the user dictated"
        assert "Assisted-by" not in json.dumps(argv)

    async def test_base_is_forwarded_only_when_given(self, repo: Path, seen: Path) -> None:
        """The default branch is gh's job (it asks GitHub); bevel never guesses it."""
        result = await create_pr(repo, base="main", gh=FIXTURE_ARGV)
        assert result.ok, result.error
        argv = rows(seen)[-1]["argv"]
        assert argv[4:6] == ["--base", "main"]

    async def test_no_base_means_gh_asks_github(self, repo: Path, seen: Path) -> None:
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        assert "--base" not in rows(seen)[-1]["argv"]

    async def test_the_child_is_told_never_to_prompt(self, repo: Path, seen: Path) -> None:
        """A gh that wants to prompt must die with a named error, not hang the loop."""
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        for row in rows(seen):
            assert row["gh_prompt_disabled"] == "1"

    async def test_the_success_report_names_the_pr_from_ghs_own_json(self, repo: Path) -> None:
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        assert "https://github.com/bevel-fixture/repo/pull/42" in result.content
        assert "#42" in result.content


# ----------------------------------------------------------------- create: refusals


class TestCreateRefusals:
    """Every refusal happens *before* the child: the fixture records nothing."""

    async def test_an_unpushed_branch_is_refused_naming_the_push(
        self, unpushed_repo: Path, seen: Path
    ) -> None:
        """The push decision, pinned: bevel never pushes. It detects and refuses.

        ``git push`` is a network write the policy engine cannot classify, and
        force-push stays out of scope entirely — m11a said so and m11b does not
        walk it back in through the side door.
        """
        result = await create_pr(unpushed_repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "git push -u origin solo" in result.error
        assert rows(seen) == [], "the refusal must happen before gh is ever spawned"

    async def test_unpushed_commits_on_a_pushed_branch_are_refused(
        self, repo: Path, seen: Path
    ) -> None:
        """The branch exists on origin but the new work does not — name that push."""
        (repo / "later.txt").write_text("later\n", encoding="utf-8")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "--no-verify", "-m", "later work")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "git push origin feature" in result.error
        assert rows(seen) == []

    async def test_a_detached_head_is_refused(self, repo: Path, seen: Path) -> None:
        git(repo, "switch", "-q", "--detach")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "detached" in result.error
        assert rows(seen) == []

    async def test_creating_a_pr_against_the_branch_you_are_on_is_refused(
        self, repo: Path, seen: Path
    ) -> None:
        git(repo, "switch", "-q", "main")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "main" in result.error
        assert rows(seen) == []

    async def test_a_repo_with_no_commits_is_refused(self, tmp_path: Path, seen: Path) -> None:
        root = tmp_path / "empty-repo"
        root.mkdir()
        result = await create_pr(root, gh=FIXTURE_ARGV)
        assert result.failed
        assert rows(seen) == []

    async def test_a_dangerous_base_never_reaches_the_child(self, repo: Path, seen: Path) -> None:
        """A base is user text and could be ``--web``; the gate runs on the
        assembled argv, not on the flags."""
        result = await create_pr(repo, base="--web", gh=FIXTURE_ARGV)
        assert result.failed
        assert "--web" in result.error
        assert rows(seen) == []


# ------------------------------------------------------------------------ auth


class TestAuth:
    async def test_the_probe_runs_before_any_pr_call(self, repo: Path, seen: Path) -> None:
        result = await auth_status(repo, gh=FIXTURE_ARGV)
        assert result.code == 0
        assert rows(seen) == [{"argv": ["auth", "status"], "gh_prompt_disabled": "1"}]

    async def test_an_unauthenticated_gh_names_the_fix_and_no_pr_is_created(
        self, repo: Path, seen: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auth is the user's, never bevel's: probe, then name the fix.

        The same shape as the git tool's missing-user.name failure: gh's own
        words, plus the command that fixes it.
        """
        mode(monkeypatch, "unauthenticated")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "gh auth login" in result.error
        assert "GH_TOKEN" in result.error
        assert "GITHUB_TOKEN" in result.error
        assert [row["argv"] for row in rows(seen)] == [["auth", "status"]], (
            "the probe ran, the create did not"
        )

    async def test_no_token_value_is_ever_printed(
        self, repo: Path, seen: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presence-only checks, the rule bevel already follows for API keys."""
        monkeypatch.setenv("GH_TOKEN", "ghp_hunter2_super_secret")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        assert "ghp_hunter2_super_secret" not in result.content
        assert "ghp_hunter2_super_secret" not in (result.error or "")
        for row in rows(seen):
            assert "ghp_hunter2_super_secret" not in json.dumps(row)


# --------------------------------------------------------------- gh failure modes


class TestGhFailureModes:
    async def test_a_gh_shaped_error_is_reported_not_swallowed(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mode(monkeypatch, "no-branch")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "no such branch" in result.error

    async def test_unparsable_json_is_a_named_failure_not_a_fake_success(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gh upgrade that reshapes its output must not read as 'PR created'."""
        mode(monkeypatch, "garbage")
        result = await create_pr(repo, gh=FIXTURE_ARGV)
        assert result.failed
        assert "json" in result.error.lower()

    async def test_a_hanging_gh_is_killed_by_the_timeout(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mode(monkeypatch, "hang")
        result = await create_pr(repo, gh=FIXTURE_ARGV, timeout_seconds=HANG_TIMEOUT)
        assert result.failed
        assert "timed out" in result.error

    async def test_the_timeout_takes_the_whole_tree(
        self, repo: Path, seen: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A grandchild is the only thing a tree kill has to prove (m10b's lesson)."""
        heartbeat = seen.parent / "heartbeat.txt"
        mode(monkeypatch, "grandchild")
        monkeypatch.setenv("BEVEL_GH_FIXTURE_HEARTBEAT", str(heartbeat))
        result = await create_pr(repo, gh=FIXTURE_ARGV, timeout_seconds=HANG_TIMEOUT)
        assert result.failed

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if heartbeat.exists():
                break
            await asyncio.sleep(0.05)
        assert heartbeat.exists(), "the grandchild never started"
        # splitlines(), not split(): each line is "<pid> <tick>", and splitting
        # on every whitespace turns the tick numbers into phantom pids. Tick 0
        # then reads as alive forever (os.kill(0, 0) signals the process group)
        # and tick 4 reads as the Windows System process -- both fail the
        # assertion for a reason that has nothing to do with the tree kill.
        pids = {int(line.split()[0]) for line in heartbeat.read_text("utf-8").splitlines() if line}
        assert pids
        for pid in pids:
            await wait_until_dead(pid)


# ---------------------------------------------------------------------- update


class TestUpdate:
    async def test_update_edits_the_title_through_the_allowlisted_surface(
        self, repo: Path, seen: Path
    ) -> None:
        result = await update_pr(repo, 7, title="Retitled", gh=FIXTURE_ARGV)
        assert result.ok, result.error
        (auth_row, edit_row) = rows(seen)
        assert auth_row["argv"] == ["auth", "status"]
        assert edit_row["argv"] == [
            "pr",
            "edit",
            "7",
            "--title",
            "Retitled",
            "--json",
            "url,number,title,state",
        ]
        assert "#7" in result.content

    async def test_update_can_edit_the_body_and_mark_ready(self, repo: Path, seen: Path) -> None:
        result = await update_pr(repo, 3, body="new body", ready=True, gh=FIXTURE_ARGV)
        assert result.ok, result.error
        argv = rows(seen)[-1]["argv"]
        assert argv[3:5] == ["--body", "new body"]
        assert "--ready" in argv

    async def test_a_non_numeric_number_is_refused_before_spawning(
        self, repo: Path, seen: Path
    ) -> None:
        result = await update_pr(repo, "7 --web", gh=FIXTURE_ARGV)
        assert result.failed
        assert "number" in result.error
        assert rows(seen) == []

    async def test_update_with_nothing_to_change_is_refused(self, repo: Path, seen: Path) -> None:
        result = await update_pr(repo, 1, gh=FIXTURE_ARGV)
        assert result.failed
        assert rows(seen) == []

    async def test_update_never_touches_the_push_check(self, unpushed_repo: Path) -> None:
        """Updating an existing PR needs no local branch state at all."""
        result = await update_pr(unpushed_repo, 1, title="t", gh=FIXTURE_ARGV)
        assert result.ok, result.error


# ------------------------------------------------------------- the module's honesty


class TestModuleHonesty:
    def test_the_docstring_says_what_the_fixture_cannot_prove(self) -> None:
        """m10e's rule: the thing that is not proven is named in the module, plainly."""
        doc = pr_module.__doc__ or ""
        assert "GitHub" in doc
        assert "cannot" in doc.lower()

    def test_the_docstring_records_the_push_decision(self) -> None:
        doc = pr_module.__doc__ or ""
        assert "push" in doc.lower()

    def test_the_docstring_records_the_tool_or_not_decision(self) -> None:
        doc = pr_module.__doc__ or ""
        assert "tool" in doc.lower()

"""Tests for the ``git`` tool (m11a).

Everything here drives a **real** git in ``tmp_path`` and never this repo --
the same rule ``test_checkpoint_snapshot.py`` follows. Mocking git would prove
only that the code agrees with our own guess about git.

The environment is pinned per test: ``GIT_CONFIG_GLOBAL`` points at an empty
file and ``GIT_CONFIG_NOSYSTEM`` is on, so a developer machine's global
``user.name`` or ``commit.gpgsign`` cannot decide whether these pass. The
identity tests depend on that isolation: the tool deliberately inherits the
user's identity rather than forging one, so "no user.name comes back as a
failure naming the fix" is only testable once the machine's config is out of
the picture.

The properties asserted most directly are the security core (an allowlist of
subcommands and options, refused *before* any spawn) and the four written-down
decisions: the trailer, the identity, branch hygiene, and the repo-root rule.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import PolicyConfig
from bevel.core.messages import ToolCall
from bevel.core.policy.engine import PolicyEngine, PolicyRequest
from bevel.core.tools.base import RiskTier, ToolContext, call_tool
from bevel.core.tools.git import MUTATING_SUBCOMMANDS, TRAILER, GitTool

_IDENTITY_ARGS = (
    "-c",
    "user.name=bevel",
    "-c",
    "user.email=bevel@localhost",
    "-c",
    "commit.gpgsign=false",
)


def write(repo: Path, name: str, text: str) -> None:
    """Write a file with no newline translation (Windows would otherwise CRLF it)."""
    target = repo / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


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
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real repository with one commit, a clean tree, and an isolated config."""
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write(root, "tracked.txt", "original\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "--no-verify", "-m", "seed")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "commit.gpgsign", "false")
    return root


@pytest.fixture
def ctx(repo: Path) -> ToolContext:
    return ToolContext(workspace=repo)


async def run(ctx: ToolContext, *command: str) -> Any:
    return await call_tool(GitTool(), {"command": list(command)}, ctx)


async def _forbid_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any child process an immediate test failure: refusals spawn nothing."""

    async def no_spawn(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"git was spawned: {args}")

    monkeypatch.setattr("asyncio.create_subprocess_exec", no_spawn)


# --------------------------------------------------------------- declaration


class TestDeclaration:
    def test_git_is_exec_unsafe_and_not_parallel_safe(self) -> None:
        """Serial: two git calls racing on the index is a corrupt index."""
        tool = GitTool()
        assert tool.risk_tier is RiskTier.EXEC_UNSAFE
        assert tool.is_parallel_safe is False

    def test_only_command_is_required_and_it_is_an_argv_list(self) -> None:
        schema = GitTool().spec.parameters
        assert schema["required"] == ["command"]
        assert schema["additionalProperties"] is False
        command = schema["properties"]["command"]
        assert command["type"] == "array"
        assert command["items"]["type"] == "string"
        assert command["minItems"] == 2

    def test_there_is_no_cwd_parameter_the_workspace_is_the_only_directory(self) -> None:
        assert "cwd" not in GitTool().spec.parameters["properties"]

    def test_the_mutating_set_covers_every_tree_changing_subcommand(self) -> None:
        assert {
            "add",
            "commit",
            "checkout",
            "switch",
            "restore",
            "reset",
            "stash",
        } <= MUTATING_SUBCOMMANDS
        assert not MUTATING_SUBCOMMANDS & {"status", "log", "diff", "show", "rev-parse"}


# ------------------------------------------------------------------------- reads


class TestReads:
    async def test_status_reports_the_workspace(self, ctx: ToolContext) -> None:
        result = await run(ctx, "git", "status")
        assert result.ok, result.error
        assert "On branch main" in result.content
        assert "[exit 0]" in result.content

    async def test_log_shows_the_seeded_commit(self, ctx: ToolContext) -> None:
        result = await run(ctx, "git", "log", "--oneline")
        assert result.ok, result.error
        assert "seed" in result.content

    async def test_rev_parse_resolves_head(self, ctx: ToolContext) -> None:
        result = await run(ctx, "git", "rev-parse", "HEAD")
        assert result.ok, result.error
        assert len(result.content.split("[exit")[0].strip()) == 40

    async def test_a_read_from_a_subdirectory_is_allowed(self, repo: Path) -> None:
        """Reads are harmless from a subdir; only mutating calls need the root."""
        sub = repo / "sub"
        sub.mkdir()
        result = await run(ToolContext(workspace=sub), "git", "status")
        assert result.ok, result.error


# ----------------------------------------------------------------------- commit


class TestCommit:
    async def test_add_then_commit_creates_a_commit(self, ctx: ToolContext, repo: Path) -> None:
        write(repo, "new.txt", "hello\n")
        added = await run(ctx, "git", "add", "--", "new.txt")
        assert added.ok, added.error
        committed = await run(ctx, "git", "commit", "-m", "Add a file")
        assert committed.ok, committed.error
        assert git(repo, "log", "-1", "--format=%s").strip() == "Add a file"

    async def test_the_trailer_is_appended_and_the_message_survives(
        self, ctx: ToolContext, repo: Path
    ) -> None:
        """The user's words are never rewritten -- only a trailer line is added."""
        write(repo, "f.txt", "x\n")
        await run(ctx, "git", "add", "--", "f.txt")
        message = "Fix the parser.\n\nBecause the tokenizer dropped quotes."
        committed = await run(ctx, "git", "commit", "-m", message)
        assert committed.ok, committed.error
        body = git(repo, "log", "-1", "--format=%B")
        assert body.startswith("Fix the parser.\n\nBecause the tokenizer dropped quotes.")
        assert TRAILER in body

    async def test_the_trailer_is_never_duplicated(self, ctx: ToolContext, repo: Path) -> None:
        """A retried commit must not stack trailers."""
        write(repo, "f.txt", "x\n")
        await run(ctx, "git", "add", "--", "f.txt")
        committed = await run(ctx, "git", "commit", "-m", f"Already credited.\n\n{TRAILER}")
        assert committed.ok, committed.error
        body = git(repo, "log", "-1", "--format=%B")
        assert body.lower().count("assisted-by: bevel") == 1

    async def test_the_author_is_the_user_not_bevel(self, ctx: ToolContext, repo: Path) -> None:
        """A commit in the user's history authored by bevel is a lie about who wrote it."""
        write(repo, "f.txt", "x\n")
        await run(ctx, "git", "add", "--", "f.txt")
        committed = await run(ctx, "git", "commit", "-m", "Authored by the user")
        assert committed.ok, committed.error
        assert git(repo, "log", "-1", "--format=%an").strip() == "Test User"
        assert git(repo, "log", "-1", "--format=%ae").strip() == "test@example.com"

    async def test_a_commit_without_user_name_fails_naming_the_fix(
        self, ctx: ToolContext, repo: Path
    ) -> None:
        git(repo, "config", "--unset-all", "user.name")
        committed = await run(ctx, "git", "commit", "-m", "no identity")
        assert committed.failed
        assert "user.name" in (committed.error or "")
        assert "git config" in (committed.error or "")

    async def test_a_commit_on_a_detached_head_is_refused(
        self, ctx: ToolContext, repo: Path
    ) -> None:
        git(repo, "checkout", "-q", "--detach")
        write(repo, "f.txt", "x\n")
        await run(ctx, "git", "add", "--", "f.txt")
        committed = await run(ctx, "git", "commit", "-m", "orphan")
        assert committed.failed
        assert "detached" in (committed.error or "")

    async def test_a_commit_without_a_message_is_refused(self, ctx: ToolContext) -> None:
        committed = await run(ctx, "git", "commit")
        assert committed.failed
        assert "-m" in (committed.error or "")

    async def test_the_commit_reports_the_branch_it_landed_on(
        self, ctx: ToolContext, repo: Path
    ) -> None:
        write(repo, "f.txt", "x\n")
        await run(ctx, "git", "add", "--", "f.txt")
        committed = await run(ctx, "git", "commit", "-m", "On main")
        assert committed.ok, committed.error
        assert "main" in committed.content

    async def test_a_failed_commit_is_reported_not_raised(self, ctx: ToolContext) -> None:
        """'nothing to commit' is an observation the model can act on, not a crash."""
        committed = await run(ctx, "git", "commit", "-m", "nothing staged")
        assert committed.ok
        assert "nothing to commit" in committed.content
        assert "[exit 1]" in committed.content


# ------------------------------------------------------------- branch hygiene


class TestBranchHygiene:
    async def test_switch_c_creates_a_new_branch(self, ctx: ToolContext, repo: Path) -> None:
        result = await run(ctx, "git", "switch", "-c", "feature")
        assert result.ok, result.error
        assert "feature" in git(repo, "branch", "--list")

    async def test_switch_c_refuses_a_name_that_already_exists(self, ctx: ToolContext) -> None:
        result = await run(ctx, "git", "switch", "-c", "main")
        assert result.failed
        assert "main" in (result.error or "")
        assert "already exists" in (result.error or "")

    async def test_branch_refuses_a_name_that_already_exists(self, ctx: ToolContext) -> None:
        result = await run(ctx, "git", "branch", "main")
        assert result.failed
        assert "already exists" in (result.error or "")

    async def test_branch_creates_a_new_named_branch(self, ctx: ToolContext, repo: Path) -> None:
        result = await run(ctx, "git", "branch", "side")
        assert result.ok, result.error
        assert "side" in git(repo, "branch", "--list")


# ------------------------------------------------------------------- refusals


class TestRefusals:
    """Every refused token, its own case, refused before anything spawns."""

    @pytest.mark.parametrize(
        "token",
        [
            "-C",
            "--git-dir",
            "--work-tree",
            "--namespace",
            "-c",
            "--exec-path",
            "--upload-pack",
            "--receive-pack",
            "--output",
            "--ext-diff",
            "--textconv",
            "--no-verify",
            "-F",
            "--file",
        ],
    )
    async def test_a_refused_option_is_named_and_nothing_spawns(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch, token: str
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await run(ctx, "git", "status", f"{token}=x")
        assert result.failed
        assert token in (result.error or "")

    async def test_a_bundled_short_config_option_is_refused(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await run(ctx, "git", "status", "-ccore.pager=evil")
        assert result.failed
        assert "-c" in (result.error or "")

    async def test_an_ext_transport_url_is_refused(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await run(ctx, "git", "status", "ext::sh -c touch% /tmp/pwn")
        assert result.failed
        assert "ext::" in (result.error or "")

    async def test_git_config_is_not_on_the_allowlist(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await run(ctx, "git", "config", "--global", "user.name", "x")
        assert result.failed
        assert "config" in (result.error or "")

    @pytest.mark.parametrize(
        "subcommand",
        ["push", "fetch", "pull", "clone", "init", "gc", "config", "worktree", "filter-branch"],
    )
    async def test_a_subcommand_off_the_allowlist_is_named(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch, subcommand: str
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await run(ctx, "git", subcommand)
        assert result.failed
        assert subcommand in (result.error or "")

    async def test_the_program_must_be_git(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _forbid_spawn(monkeypatch)
        result = await call_tool(GitTool(), {"command": ["python", "status"]}, ctx)
        assert result.failed
        assert "python" in (result.error or "")

    async def test_a_mutating_subcommand_from_a_subdirectory_is_refused(self, repo: Path) -> None:
        sub = repo / "sub"
        sub.mkdir()
        result = await run(ToolContext(workspace=sub), "git", "add", ".")
        assert result.failed
        assert "repository root" in (result.error or "")


# ------------------------------------------------------ policy-engine seam


class TestPolicySeam:
    """The argv-as-``command`` shape is what makes the built-in rules apply.

    ``git status`` is free and ``git reset --hard`` is destructive *because*
    the engine reads ``arguments["command"]`` for an EXEC-tier tool. A git tool
    that hid its argv behind another property name would get none of that.
    """

    def _request(self, command: list[str]) -> PolicyRequest:
        call = ToolCall(id="c1", name="git", arguments="{}")
        return PolicyRequest(call=call, tool=GitTool(), name="git", arguments={"command": command})

    async def test_git_status_is_free(self, ctx: ToolContext, tmp_path: Path) -> None:
        engine = PolicyEngine(ctx, config=PolicyConfig(db_path=tmp_path / "policy.db"))
        try:
            verdict = await engine(self._request(["git", "status"]))
        finally:
            engine.close()
        assert verdict.decision.value == "allow"
        assert "git status" in verdict.reason

    async def test_git_reset_hard_is_asked_as_destructive(
        self, ctx: ToolContext, tmp_path: Path
    ) -> None:
        engine = PolicyEngine(ctx, config=PolicyConfig(db_path=tmp_path / "policy.db"))
        try:
            verdict = await engine(self._request(["git", "reset", "--hard"]))
        finally:
            engine.close()
        assert verdict.decision.value == "deny"  # DenyAllApprover: nobody was asked
        assert "destructive" in verdict.reason

    async def test_an_empty_argv_is_refused_as_unclassifiable(
        self, ctx: ToolContext, tmp_path: Path
    ) -> None:
        engine = PolicyEngine(ctx, config=PolicyConfig(db_path=tmp_path / "policy.db"))
        try:
            verdict = await engine(self._request([]))
        finally:
            engine.close()
        assert verdict.decision.value == "deny"
        assert "did not supply a command" in verdict.reason

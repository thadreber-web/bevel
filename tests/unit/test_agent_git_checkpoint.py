"""A mutating ``git`` call takes the session checkpoint, a read does not (m11a).

The loop's rule (AGENTS.md, providers-tools instructions) is that anything which
mutates the filesystem is preceded by a checkpoint, and checkpoints run even
under ``--yolo``. A ``git commit`` rewrites the tree a checkpoint ref describes,
so it must trigger the same snapshot the first ``write`` does — otherwise the
one operation that is hardest to undo is the one with no way back.

The decision, pinned here: the checkpoint fires on a *mutating* git subcommand
(add, commit, checkout, switch, restore, reset, stash, ...) and not on a read
(status, log, diff, ...). It is decided from the tool and its arguments before
execution, so a commit that later fails still cost a checkpoint — the
conservative direction.

Driven through the real loop against real git in ``tmp_path``, never this repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from bevel.config.schema import PolicyConfig, PolicyMode
from bevel.core.agent import Agent, AgentEventType
from bevel.core.checkpoint import checkpoint_ref
from bevel.core.policy.approvals import AutoApprover
from bevel.core.tools.git import GitTool
from bevel.providers.base import Capabilities
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

_IDENTITY_ARGS = (
    "-c",
    "user.name=bevel",
    "-c",
    "user.email=bevel@localhost",
    "-c",
    "commit.gpgsign=false",
)


def _git(repo: Path, *args: str) -> str:
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


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "seed.txt").write_text("seed\n", encoding="utf-8", newline="")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "--no-verify", "-m", "seed")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "commit.gpgsign", "false")
    return root


def _agent(repo: Path, tmp_path: Path, turns: list[ScriptedTurn]) -> Agent:
    provider = MockProvider(turns=turns)
    provider.capabilities = Capabilities(supports_tools=True, supports_streaming=True)
    return Agent(
        provider,
        repo,
        tools=[GitTool()],
        approver=AutoApprover(),
        policy_config=PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db"),
    )


def _ask(name: str, **arguments: object) -> ScriptedTurn:
    return ScriptedTurn(content="", tool_calls=(tool_call("c1", name, arguments),))


async def test_a_commit_takes_the_checkpoint_before_it_runs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "new.txt").write_text("hello\n", encoding="utf-8", newline="")
    agent = _agent(
        repo,
        tmp_path,
        [
            _ask("git", command=["git", "add", "--", "new.txt"]),
            _ask("git", command=["git", "commit", "-m", "Add a file"]),
            ScriptedTurn(content="done"),
        ],
    )

    result = await agent.run("commit the new file")

    checkpoints = [e for e in result.events if e.type is AgentEventType.CHECKPOINT]
    assert checkpoints, "a mutating git call must trigger the session checkpoint"
    # The commit really happened, and the checkpoint ref exists to undo it.
    assert _git(repo, "log", "-1", "--format=%s").strip() == "Add a file"
    listed = _git(repo, "for-each-ref", "refs/heads/bevel/checkpoint/").strip()
    assert listed, "the checkpoint ref was never created"


async def test_a_read_does_not_take_a_checkpoint(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    agent = _agent(
        repo,
        tmp_path,
        [_ask("git", command=["git", "status"]), ScriptedTurn(content="done")],
    )

    result = await agent.run("what is the status")

    assert not [e for e in result.events if e.type is AgentEventType.CHECKPOINT]
    assert not _git(repo, "for-each-ref", "refs/heads/bevel/checkpoint/").strip()
    assert not (repo / checkpoint_ref("x")).exists()

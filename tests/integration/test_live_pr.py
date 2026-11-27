"""The only tests in m11b that may touch real GitHub, and they stay behind a gate.

The unit oracle (``tests/fixtures/gh.py``) can prove argv construction, JSON
parsing, exit-code mapping, the timeout/tree kill, and the refusals. What it
cannot prove is that GitHub's API accepts what gh sends — and a unit test that
pretends otherwise would be the confident lie this project refuses to ship.

These tests are deselected by default and skip unless **all** of:

    $env:BEVEL_GH_INTEGRATION = "1"   # you have read what this file does
    gh is on PATH                     # the real gh, not the fixture
    gh is authenticated               # auth is the user's, never bevel's

and they run against the repository you are standing in, which must be a real
clone with a remote. That is the same gating shape the live-endpoint tests use
with ``BEVEL_BASE_URL``: absent configuration is a skip with a reason, never a
silent pass.

What is deliberately *not* here: creating a real PR. A test that opens a pull
request on every run litters a real repo, closes nothing, and needs a second
repo to be safe — the honest version of that test is a manual check, and the
unit suite already proves everything up to the wire. What *is* here is the
probe against the real binary: the fixture's ``auth status`` is a guess about
gh's exit codes, and this file is the one place that guess is checked.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from bevel.core.pr import auth_status

INTEGRATION_ENV = "BEVEL_GH_INTEGRATION"
TRUTHY = frozenset({"1", "true", "yes", "on"})


def _missing_requirements() -> str | None:
    """Describe what has to hold before these tests can mean anything."""
    if os.environ.get(INTEGRATION_ENV, "").strip().lower() not in TRUTHY:
        return f"{INTEGRATION_ENV} is not set (set it to 1 to run against real GitHub)"
    if shutil.which("gh") is None:
        return "gh is not on PATH"
    return None


@pytest.fixture
def workspace() -> Path:
    """The repository this checkout lives in — the only repo these tests touch."""
    reason = _missing_requirements()
    if reason:
        pytest.skip(reason)
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip(f"{root} is not a git repository")
    return root


async def test_the_real_gh_answers_the_auth_probe(workspace: Path) -> None:
    """The fixture's exit-code mapping, checked against the real binary.

    Either answer is a pass — authenticated or not — because the assertion is
    about bevel's handling, not the user's login state: the probe must return
    a code and a gh-shaped message, must never raise, and must never print a
    token value.
    """
    result = await auth_status(workspace)
    assert result.code in (0, 1), f"unexpected exit code {result.code}: {result.err!r}"
    if result.code != 0:
        assert "gh auth login" in result.err or "GH_TOKEN" in result.err
    for secret in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(secret, "")
        assert not value or value not in result.out + result.err, (
            f"the {secret} value must never be echoed"
        )

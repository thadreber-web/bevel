"""Shared fixtures for the Pilot tests.

Two things must be true for these tests to be honest:

* they drive the real :class:`~bevel.tui.app.BevelApp` through Textual's Pilot
  against :class:`~bevel.providers.mock.MockProvider` -- no live endpoint, ever;
* they never touch the real ``~/.config/bevel`` or ``~/.cache/bevel``. The policy
  engine opens SQLite at ``policy_db_path()`` unless the config names a path, so
  the homes are redirected *and* the database is pinned into ``tmp_path``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from bevel.config import loader
from bevel.config.schema import PolicyConfig, PolicyMode
from bevel.providers.mock import MockProvider, ScriptedTurn
from bevel.tui.app import BevelApp, RunMode


@pytest.fixture(autouse=True)
def isolated_homes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep config, cache and policy state out of the real user directories."""
    monkeypatch.setattr(loader, "config_home", lambda: tmp_path / "config")
    monkeypatch.setattr(loader, "cache_home", lambda: tmp_path / "cache")


@pytest.fixture
def make_app(tmp_path: Path, isolated_homes: None) -> Callable[..., BevelApp]:
    """Build an app over a throwaway workspace driven by a scripted provider."""

    def factory(*turns: ScriptedTurn, mode: RunMode = RunMode.AUTO, delay: float = 0.0) -> BevelApp:
        policy = PolicyConfig(mode=PolicyMode.AUTO, db_path=tmp_path / "policy.db")
        provider = MockProvider(turns=list(turns), delay=delay)
        return BevelApp(provider=provider, workspace=tmp_path, policy=policy, mode=mode)

    return factory

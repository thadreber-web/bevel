"""What the loop is allowed to claim it does not do.

``UNBUILT_SEAMS`` is the honest-disclaimer list: it rides on every ``AgentResult``
and into the system prompt, so a stale entry is a lie about a built feature and a
missing one is a lie about an unbuilt one. The live integration suite asserts this
too, but it skips whenever no endpoint is configured, which is most machines --
including CI. These assertions must therefore hold with no model in sight.
"""

from __future__ import annotations

import pytest

from bevel.core.agent import DEFAULT_SYSTEM_PROMPT, UNBUILT_SEAMS


def test_unbuilt_seams_is_context_only() -> None:
    assert len(UNBUILT_SEAMS) == 2, UNBUILT_SEAMS
    assert all("m08" in seam for seam in UNBUILT_SEAMS), UNBUILT_SEAMS


@pytest.mark.parametrize("stale", ["m05", "policy", "approval", "m06", "checkpoint"])
def test_shipped_features_are_no_longer_reported_as_unbuilt(stale: str) -> None:
    """m05 and m06 shipped; the loop must stop disclaiming them.

    A seam that outlives its milestone is worse than no seam at all: the model is
    told there is no policy engine (or no way back) and stops expecting tool calls
    to be refused, or a run to be undoable.
    """
    assert not any(stale in seam.lower() for seam in UNBUILT_SEAMS), (
        f"{stale!r} still appears in UNBUILT_SEAMS: {UNBUILT_SEAMS}"
    )


def test_system_prompt_no_longer_denies_a_policy_engine() -> None:
    lowered = DEFAULT_SYSTEM_PROMPT.lower()
    assert "no policy engine" not in lowered, DEFAULT_SYSTEM_PROMPT
    assert "policy is enforced" in lowered, DEFAULT_SYSTEM_PROMPT


def test_system_prompt_no_longer_denies_checkpointing() -> None:
    lowered = DEFAULT_SYSTEM_PROMPT.lower()
    assert "no checkpointing" not in lowered, DEFAULT_SYSTEM_PROMPT
    assert "snapshotted" in lowered, DEFAULT_SYSTEM_PROMPT

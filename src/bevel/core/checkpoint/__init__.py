"""Session checkpoints (milestone m06).

One snapshot of the working tree per session, taken before the agent's first
write, and a restore back to it. See :mod:`bevel.core.checkpoint.snapshot` for
why the snapshot is built from a throwaway index rather than a git worktree.

Checkpoints run even under ``--yolo``: auto-approval governs what the agent may
do, not whether it can be undone.
"""

from __future__ import annotations

from bevel.core.checkpoint.snapshot import (
    CheckpointResult,
    RestoreResult,
    checkpoint_ref,
    list_checkpoints,
    restore,
    snapshot,
)

__all__ = [
    "CheckpointResult",
    "RestoreResult",
    "checkpoint_ref",
    "list_checkpoints",
    "restore",
    "snapshot",
]

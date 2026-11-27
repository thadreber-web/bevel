"""The one place bevel spawns ``git``.

There were three. :func:`bevel.core.checkpoint.snapshot._git` grew a tree kill
and an identity env for the m06 snapshot; :func:`bevel.core.memory.store._origin_url`
grew a second copy of the same dance for the m09a scope probe; and m11a's
``git`` tool would have been the third. A tree kill is not something to
reimplement three times -- that reasoning is already written into
:mod:`bevel.core._process`, which exists for exactly the same reason.

The contract every caller depends on, and the one that makes a shared runner
safe to share:

* **never raises.** A machine with no git, a repo with no commits, a wedged
  index lock: all come back as a non-zero :class:`GitResult`. Callers read
  ``code``; nobody has to remember a try/except.
* **always bounded.** A timeout kills the whole process tree through
  :class:`~bevel.core._process.TreeKiller`, so a ``git`` stuck on a lock file
  cannot outlive the call.
* **never inherits a caller's whims.** The child gets the parent environment
  plus whatever ``extra_env`` says. No caller can hand it a model-controlled
  environment -- that refusal belongs to the tool that receives model output,
  and :mod:`bevel.core.tools.git` owns it.

``cwd`` is always the workspace: a caller that wants a different directory is
asking the wrong question, because bevel's whole confinement story is that the
workspace is the only directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bevel.core._process import TreeKiller, close_process, uses_new_session

__all__ = ["DEFAULT_GIT_TIMEOUT_SECONDS", "GitResult", "run_git"]

#: How long one git call may take before its tree is killed.
DEFAULT_GIT_TIMEOUT_SECONDS: Final[float] = 30.0
#: How long to let the pipes drain after a kill, so output written before the
#: kill is still reported rather than thrown away with the process.
_DRAIN_GRACE_SECONDS: Final[float] = 2.0


@dataclass(frozen=True)
class GitResult:
    """One git call's exit code and output.

    ``code`` mirrors git where git ran, and uses the shell's own convention
    where it did not: ``127`` when the child could not be started at all,
    ``124`` when it was killed for running too long. Those numbers are not
    invented -- they are what ``sh`` reports for those two cases, and a caller
    that has to distinguish "no git" from "git said no" can.
    """

    code: int
    out: str
    err: str


async def run_git(
    workspace: Path | str,
    args: Sequence[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> GitResult:
    """Run one git command in ``workspace``. Never raises; read ``code``.

    ``args`` is the argv *after* the program name, and is passed straight to
    :func:`asyncio.create_subprocess_exec` as its own list -- never through a
    shell. Validating what may appear in it is the caller's job, and
    :mod:`bevel.core.tools.git` is the caller that has to do it carefully.
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
        return GitResult(127, "", f"cannot run git: {exc}")

    killer = TreeKiller(proc.pid)
    try:
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            killer.kill()
            with contextlib.suppress(TimeoutError, OSError):
                await asyncio.wait_for(proc.communicate(), timeout=_DRAIN_GRACE_SECONDS)
            return GitResult(124, "", f"git {' '.join(args)} timed out")
        return GitResult(
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

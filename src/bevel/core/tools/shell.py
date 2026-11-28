"""The ``shell`` tool: run a command, with a timeout that really stops it.

Two rules from AGENTS.md shape everything here.

**Never ``shell=True``.** ``command`` is an argv array and stays one all the way
to :func:`asyncio.create_subprocess_exec`. Routing through ``cmd.exe`` would
inherit its quoting rules, which is a command-injection surface dressed up as
convenience -- and the model is the one typing the string.

**Kill trees, not parents.** A timeout that only kills the direct child leaves
grandchildren alive and holding the pipes open, so the next call hangs. The
mechanism lives in :mod:`bevel.core._process`, shared with the checkpoint
module, because a tree kill is not something to reimplement twice.

The tier is ``EXEC_UNSAFE`` and nothing better: whether a particular argv is
safe is a decision about the *command*, made by a rule, and rules are m05.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from bevel.core._process import TreeKiller, close_process, uses_new_session
from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.types import ToolSpec

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
MAX_TIMEOUT_SECONDS: Final[float] = 600.0
DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 32_000
MAX_OUTPUT_BYTES_CAP: Final[int] = 1_000_000
#: Bytes taken from a pipe per read.
_CHUNK_BYTES: Final[int] = 65_536
#: How long to let the pipes drain after a kill, so output produced *before*
#: the timeout is not thrown away with the process.
_DRAIN_GRACE_SECONDS: Final[float] = 2.0


@dataclass
class _Stream:
    """One captured pipe: the kept bytes, and whether more existed."""

    chunks: list[bytes] = field(default_factory=list)
    kept: int = 0
    total: int = 0
    cap: int = DEFAULT_MAX_OUTPUT_BYTES

    @property
    def truncated(self) -> bool:
        return self.total > self.kept

    def accept(self, chunk: bytes) -> None:
        """Record a chunk, keeping only the first ``cap`` bytes of the stream.

        Reading continues past the cap (discarding) rather than stopping,
        because a pipe nobody drains keeps the writer blocked and the child
        unwaitable.
        """
        self.total += len(chunk)
        if self.kept >= self.cap:
            return
        room = self.cap - self.kept
        self.chunks.append(chunk[:room])
        self.kept += min(len(chunk), room)

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


class ShellTool:
    """Run a command in the workspace and capture its output."""

    spec: Final[ToolSpec] = ToolSpec(
        name="shell",
        description=(
            "Run a command and return its stdout, stderr and exit code. `command` is "
            "an argv array, never a shell string: put each argument in its own "
            "element. The command runs in the workspace with a timeout, and a "
            "timeout kills the whole process tree. A non-zero exit code is reported, "
            "not treated as a tool failure."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "array",
                    "description": "The program and its arguments, one array element each.",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Working directory, relative to the workspace (default: the workspace)."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Seconds before the command is killed (default {DEFAULT_TIMEOUT_SECONDS:g})."
                    ),
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_SECONDS,
                    "default": DEFAULT_TIMEOUT_SECONDS,
                },
                "max_output_bytes": {
                    "type": "integer",
                    "description": (
                        f"Bytes kept from each output stream (default {DEFAULT_MAX_OUTPUT_BYTES})."
                    ),
                    "minimum": 1,
                    "maximum": MAX_OUTPUT_BYTES_CAP,
                    "default": DEFAULT_MAX_OUTPUT_BYTES,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.EXEC_UNSAFE
    is_parallel_safe: Final[bool] = False

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        argv = [str(item) for item in arguments["command"]]
        if not argv[0].strip():
            raise ToolError("'command' must name a program to run")

        cwd = self._working_directory(arguments, ctx)
        timeout = float(arguments.get("timeout", DEFAULT_TIMEOUT_SECONDS))
        cap = min(
            int(arguments.get("max_output_bytes", DEFAULT_MAX_OUTPUT_BYTES)), MAX_OUTPUT_BYTES_CAP
        )

        proc = await _spawn(_spawn_argv(ctx.sandbox_prefix, argv), os.fspath(cwd))
        killer = TreeKiller(proc.pid)
        stdout = _Stream(cap=cap)
        stderr = _Stream(cap=cap)
        timed_out = False
        try:
            timed_out = await _run(proc, stdout, stderr, timeout, killer)
        finally:
            killer.close()
            close_process(proc)

        body = _render(stdout, stderr, proc.returncode)
        if timed_out:
            return ToolResult(
                ok=False,
                content=body,
                error=(
                    f"shell: timed out after {timeout:g}s running {argv[0]!r}; "
                    "the process tree was killed"
                ),
            )
        return ToolResult(ok=True, content=body)

    def _working_directory(
        self, arguments: Mapping[str, Any], ctx: ToolContext
    ) -> os.PathLike[str]:
        if "cwd" not in arguments:
            return ctx.workspace
        cwd = ctx.require_within_workspace(str(arguments["cwd"]))
        if not cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {ctx.display(cwd)}")
        return cwd


def _spawn_argv(prefix: tuple[str, ...], argv: list[str]) -> list[str]:
    """The argv actually spawned: the bwrap prefix (m12a), then the command.

    An empty prefix -- POLICY_ONLY, or a directly-constructed context in a
    test -- leaves the argv byte-for-byte what the model asked for. The
    prefix is bevel-built from the verified backend and never model-built;
    the model's argv lands after the ``--`` separator, so it can never be
    parsed as a bwrap option.
    """
    return [*prefix, *argv] if prefix else list(argv)


async def _spawn(argv: list[str], cwd: str) -> asyncio.subprocess.Process:
    """Start the child, in its own process group where that exists.

    ``start_new_session`` is POSIX-only -- ``subprocess.Popen`` raises
    ``ValueError`` for it on Windows -- so it is passed conditionally. Windows
    gets its tree kill from the Job Object instead.
    """
    kwargs: dict[str, Any] = {
        "stdin": asyncio.subprocess.DEVNULL,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
    }
    if uses_new_session():
        kwargs["start_new_session"] = True
    try:
        return await asyncio.create_subprocess_exec(*argv, **kwargs)
    except FileNotFoundError as exc:
        raise ToolError(f"program not found: {argv[0]!r}") from exc
    except PermissionError as exc:
        raise ToolError(f"permission denied running {argv[0]!r}: {exc}") from exc
    except OSError as exc:
        raise ToolError(f"could not run {argv[0]!r}: {type(exc).__name__}: {exc}") from exc


async def _run(
    proc: asyncio.subprocess.Process,
    stdout: _Stream,
    stderr: _Stream,
    timeout_seconds: float,
    killer: TreeKiller,
) -> bool:
    """Wait for the command; kill its tree on timeout. Returns whether it timed out."""
    tasks = {
        asyncio.ensure_future(_drain(proc.stdout, stdout)),
        asyncio.ensure_future(_drain(proc.stderr, stderr)),
        asyncio.ensure_future(proc.wait()),
    }
    _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    if not pending:
        return False

    killer.kill()
    # The pipes only reach EOF once the whole tree is dead, so give the kernel a
    # moment to close them; anything already written is still reported.
    await asyncio.wait(pending, timeout=_DRAIN_GRACE_SECONDS)
    for task in pending:
        task.cancel()
    return True


async def _drain(stream: asyncio.StreamReader | None, sink: _Stream) -> None:
    """Read a pipe to EOF."""
    if stream is None:  # pragma: no cover - both pipes are always created
        return
    while True:
        chunk = await stream.read(_CHUNK_BYTES)
        if not chunk:
            return
        sink.accept(chunk)


def _render(stdout: _Stream, stderr: _Stream, exit_code: int | None) -> str:
    parts: list[str] = []
    if stdout.text().strip():
        parts.append(stdout.text().rstrip("\n"))
    if stdout.truncated:
        parts.append(f"[stdout truncated at {stdout.cap} bytes]")
    if stderr.text().strip():
        parts.append(f"[stderr]\n{stderr.text().rstrip(chr(10))}")
    if stderr.truncated:
        parts.append(f"[stderr truncated at {stderr.cap} bytes]")
    parts.append(f"[exit {exit_code if exit_code is not None else 'unknown'}]")
    return "\n".join(parts)


__all__ = ["ShellTool"]

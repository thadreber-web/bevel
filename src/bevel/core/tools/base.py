"""The tool seam: declaration, context, result, dispatch.

A tool is a frozen declaration (name, JSON Schema, risk tier, parallel-safety)
plus an ``async execute`` that receives already-parsed arguments and a
:class:`ToolContext`. Tools never read config, never touch ``os.environ``, and
never reach for a global: their entire world is the arguments and the context
they were handed. That is what makes a tool callable from the TUI, from the
headless loop, and from an eval without any of them behaving differently.

Risk tiers are *declared* here and *enforced* by the policy engine in m05.
Declaring them honestly now is the precondition for that engine existing --
an engine that trusts a declaration can only work if the declaration was never
inflated to get a call through.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from bevel.core.tools.schema import validate
from bevel.core.tools.types import ToolSpec

if TYPE_CHECKING:
    from bevel.core.memory.store import MemoryStore
    from bevel.core.session.store import TodoStore


class RiskTier(StrEnum):
    """How much damage a call could do, and therefore what it costs to get one.

    The ladder is fixed by the plan; the behaviour column is not implemented
    here. m05 evaluates against it, m12 enforces the parts the kernel can
    enforce. What this module owns is the *declaration*, and a tool that
    understates its own tier is a bug report waiting to happen.

    +---------------------------+------------------------------------------+
    | tier                      | behaviour                                |
    +===========================+==========================================+
    | ``READ``                  | free                                     |
    +---------------------------+------------------------------------------+
    | ``WRITE_WORKSPACE``       | ask, or auto in ``auto`` mode            |
    +---------------------------+------------------------------------------+
    | ``WRITE_OUTSIDE``         | always ask                               |
    +---------------------------+------------------------------------------+
    | ``EXEC_SAFE``             | free once allowlisted                    |
    +---------------------------+------------------------------------------+
    | ``EXEC_UNSAFE``           | ask                                      |
    +---------------------------+------------------------------------------+
    | ``EXEC_DESTRUCTIVE``      | always ask + typed confirmation          |
    +---------------------------+------------------------------------------+
    | ``NETWORK_EGRESS``        | ask                                      |
    +---------------------------+------------------------------------------+

    The m03 tools produce ``READ``, ``WRITE_WORKSPACE``, and ``EXEC_UNSAFE``;
    m09a's ``remember`` produces ``WRITE_OUTSIDE``, since it writes to the
    user-level memory database rather than the workspace.
    The rest have no producer yet, deliberately: ``EXEC_SAFE`` and
    ``EXEC_DESTRUCTIVE`` come from matching a *specific* argv against rules
    (m05), and ``NETWORK_EGRESS`` from the sandbox (m12). A tool cannot claim
    ``EXEC_SAFE`` on its own -- safety is a property of the command, decided by a rule, not of
    the tool that runs it. That is why :class:`~bevel.core.tools.shell` declares
    ``EXEC_UNSAFE`` and nothing better.
    """

    READ = "read"
    WRITE_WORKSPACE = "write_workspace"
    WRITE_OUTSIDE = "write_outside"
    EXEC_SAFE = "exec_safe"
    EXEC_UNSAFE = "exec_unsafe"
    EXEC_DESTRUCTIVE = "exec_destructive"
    NETWORK_EGRESS = "network_egress"


class ToolError(Exception):
    """A tool refused a call, or could not carry it out.

    Always actionable: the message names the thing that failed and, where there
    is a fix, states it. Reaching a milestone that has not been built yet is
    also a ``ToolError`` -- refusing loudly beats pretending.
    """


@dataclass(frozen=True)
class ToolResult:
    """What a tool hands back to the loop.

    Tools return this instead of raising, because a failed tool call is a
    normal event in an agent conversation: the model reads the error and tries
    again. An exception would abort the iteration; a result keeps it in the
    transcript. ``error`` is set exactly when ``ok`` is false.
    """

    ok: bool
    content: str = ""
    error: str | None = None

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, error=error)

    @property
    def failed(self) -> bool:
        return not self.ok


@dataclass(frozen=True)
class ToolContext:
    """The world a tool is allowed to see.

    One workspace, resolved once, and the path rules that hang off it. Every
    filesystem tool funnels its arguments through
    :meth:`require_within_workspace`, which is the T0 path jail: it is not
    security (a sandbox is security, and that is m12), it is the difference
    between "the agent edited the repo" and "the agent edited your home
    directory" after a model hallucinates a path.
    """

    workspace: Path
    #: Long-term memory, or ``None`` when it is not configured. Optional and
    #: last so that every existing ``ToolContext(workspace)`` call is unchanged.
    memory: MemoryStore | None = None
    #: The session's todo list, or ``None`` when no session store is attached.
    #: Optional and last, for the same reason as ``memory``.
    todos: TodoStore | None = None
    #: The bwrap argv prefix (m12a) the shell tool prepends to every spawn,
    #: verified by the self-test at startup. Empty means POLICY_ONLY: the
    #: spawn is unprefixed and the path jail below is the only boundary.
    #: Optional and last, for the same reason as ``memory``.
    sandbox_prefix: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Resolving here rather than per call: ``resolve()`` follows symlinks
        # and expands Windows 8.3 short names, so doing it once at the boundary
        # means every containment test below compares canonical paths.
        object.__setattr__(
            self, "workspace", Path(self.workspace).expanduser().resolve(strict=False)
        )

    def resolve(self, path: str) -> Path:
        """Absolute, symlink-resolved form of ``path``; relative to the workspace."""
        candidate = Path(os.path.expanduser(path))
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        return candidate.resolve(strict=False)

    def require_within_workspace(self, path: str) -> Path:
        """Resolve ``path`` and refuse it if it lands outside the workspace.

        Fails closed. Out-of-workspace access is not a capability any m03 tool
        has: it needs an approval, and approvals are m05.
        """
        resolved = self.resolve(path)
        if _within(self.workspace, resolved):
            return resolved
        raise ToolError(
            f"{path!r} resolves to {resolved}, which is outside the workspace "
            f"{self.workspace}. bevel confines every tool to the workspace until "
            "m05 (policy and approvals) can approve an out-of-workspace path."
        )

    def display(self, path: Path) -> str:
        """Short form for messages: workspace-relative when that is true."""
        return self.relative(path) if _within(self.workspace, path) else str(path)

    def relative(self, path: Path) -> str:
        """Workspace-relative path with forward slashes, for stable output."""
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError:  # pragma: no cover - guarded by callers
            return path.as_posix()


def _within(root: Path, candidate: Path) -> bool:
    """Is ``candidate`` at or below ``root``?

    ``normcase`` because Windows paths are case-insensitive and a comparison
    that ignores that would refuse ``D:\\Dev\\Harness\\a.py`` as "outside"
    ``D:\\dev\\harness``.
    """
    root_str = os.path.normcase(str(root)).rstrip(os.sep)
    candidate_str = os.path.normcase(str(candidate))
    return candidate_str == root_str or candidate_str.startswith(root_str + os.sep)


@runtime_checkable
class Tool(Protocol):
    """A capability the model can be offered.

    Implementations are stateless values: the same instance serves every call,
    so a tool can be shared across concurrent reads without a lock.

    The three members are declared as read-only properties rather than plain
    attributes so that a tool may declare them as ``Final`` class attributes --
    they are constants, and saying so is the point -- which mypy will not accept
    against a mutable protocol attribute.
    """

    @property
    def spec(self) -> ToolSpec: ...

    @property
    def risk_tier(self) -> RiskTier: ...

    @property
    def is_parallel_safe(self) -> bool: ...

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """Run the call. ``arguments`` has already passed schema validation."""
        ...


async def call_tool(tool: Tool, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
    """Validate, execute, and never raise into the loop.

    Validation happens here rather than in each tool so that "arguments were
    checked against the declared schema before anything ran" is a property of
    the seam, not a convention each tool remembers. The model gets every
    violation at once -- one re-ask can fix all of them.
    """
    violations = validate(arguments, tool.spec.parameters)
    if violations:
        details = "; ".join(str(violation) for violation in violations)
        return ToolResult.failure(f"{tool.spec.name}: arguments failed validation: {details}")

    try:
        return await tool.execute(arguments, ctx)
    except ToolError as exc:
        return ToolResult.failure(f"{tool.spec.name}: {exc}")
    except OSError as exc:
        # An unexpected filesystem error is still an observation the model can
        # act on, so it is reported with its type -- ``PermissionError: [WinError 5]``
        # and ``FileNotFoundError`` call for very different corrections.
        return ToolResult.failure(f"{tool.spec.name}: {type(exc).__name__}: {exc}")


__all__ = [
    "RiskTier",
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "call_tool",
]

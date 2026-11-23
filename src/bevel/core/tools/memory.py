"""The ``remember`` and ``recall`` tools: the model's own long-term memory.

Two tools, one for each direction. ``recall`` is a plain read and is
parallel-safe. ``remember`` writes to ``~/.local/share/bevel/memories.db`` --
outside the workspace, permanently, in a file the user cannot undo by checking
out a branch -- so it declares ``WRITE_OUTSIDE``.

That tier means *always ask*, and it means that here rather than in the brief:
the policy engine returns an ask for ``WRITE_OUTSIDE`` before it ever looks at
``PolicyMode.AUTO``, so ``remember`` is not auto-approved in auto mode. The
alternative -- declaring it ``READ`` to get out of the way -- would be exactly
the inflated declaration the tier ladder exists to prevent.

The one exception is ``[memory] auto_accept = true``, which lets the engine
allow this tool by name. It exists because the model initiates these writes
itself and learned rules cannot cover them: they are keyed on argv, and
``remember`` has none, so ``a`` at the approval prompt has nothing to learn.
The exemption is off by default, names only ``remember``, and does not reach a
call whose path escapes the workspace jail.

Both tools take the store from :attr:`ToolContext.memory`. The registry holds
store-less instances so the tool list is stable whether or not memory is
configured; a call without a store fails loudly instead of silently doing
nothing, which is the difference between "memory is off" and "memory is broken
and nobody noticed".
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from bevel.core.memory.recall import recall
from bevel.core.memory.store import MemoryType, repo_scope
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import ProviderError

if TYPE_CHECKING:
    from bevel.core.memory.store import MemoryStore

__all__ = ["RecallTool", "RememberTool", "memory_tools"]

_TYPE_VALUES: Final[list[str]] = ["fact", "feedback", "decision", "reference"]


class RememberTool:
    """Store one thing worth knowing, permanently."""

    spec: Final[ToolSpec] = ToolSpec(
        name="remember",
        description=(
            "Save a durable fact, piece of feedback, decision, or reference to bevel's "
            "long-term memory. Use it for things that stay true across sessions: a "
            "project convention, a user preference, a decision and its reason. It "
            "writes outside the workspace and is never overwritten; correcting an "
            "earlier memory means remembering the new one and superseding the old."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The thing to remember, stated once and without context.",
                    "minLength": 1,
                    "maxLength": 4000,
                },
                "type": {
                    "type": "string",
                    "enum": _TYPE_VALUES,
                    "description": (
                        "What kind of thing it is. 'decision' and 'feedback' outrank the "
                        "others when recalling, so use them when they fit."
                    ),
                },
            },
            "required": ["content", "type"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.WRITE_OUTSIDE
    is_parallel_safe: Final[bool] = False

    def __init__(self, store: MemoryStore | None = None, workspace: Path | None = None) -> None:
        self._store = store
        self._workspace = workspace

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        store = self._store or ctx.memory
        if store is None:
            return ToolResult.failure(
                "remember: long-term memory is not configured for this session"
            )
        workspace = self._workspace or ctx.workspace
        scope = await repo_scope(Path(workspace))
        content = str(arguments["content"]).strip()
        type_ = MemoryType(str(arguments["type"]))

        vector: tuple[float, ...] | None = None
        vector_model: str | None = None
        note = ""
        embedder = store.embedder
        if (
            store.config.vectors
            and embedder is not None
            and embedder.capabilities.supports_embeddings
        ):
            try:
                vector = await embedder.embed(content)
                vector_model = embedder.model
            except ProviderError as exc:
                # Losing the vector must not lose the memory: the text is the
                # part the user asked for, and lexical recall still finds it.
                vector = None
                vector_model = None
                note = f" (stored without a vector: the /embeddings endpoint failed: {exc})"
        elif store.config.vectors and embedder is None:
            note = " (stored without a vector: no embedding model is configured)"

        memory = store.add(
            content=content, type=type_, scope=scope, vector=vector, vector_model=vector_model
        )
        return ToolResult(ok=True, content=f"remembered #{memory.id} [{type_}] for {scope}{note}")


class RecallTool:
    """Ask memory what it knows about a question."""

    spec: Final[ToolSpec] = ToolSpec(
        name="recall",
        description=(
            "Search bevel's long-term memory for what it knows about a question. "
            "Returns the stored facts, feedback, decisions, and references that match, "
            "ranked so that decisions and feedback come first. Read-only and cheap; "
            "ask before assuming you already know a project convention."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in the words you would search with.",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.READ
    is_parallel_safe: Final[bool] = True

    def __init__(self, store: MemoryStore | None = None, workspace: Path | None = None) -> None:
        self._store = store
        self._workspace = workspace

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        store = self._store or ctx.memory
        if store is None:
            return ToolResult.failure("recall: long-term memory is not configured for this session")
        workspace = self._workspace or ctx.workspace
        result = await recall(store, str(arguments["query"]), workspace=Path(workspace))
        if not result.memories:
            return ToolResult(
                ok=True,
                content=f"no memory matched {arguments['query']!r} ({result.mode} recall)."
                + (f" {'; '.join(result.notes)}" if result.notes else ""),
            )
        lines = [f"{result.mode} recall, {len(result.memories)} of {store.config.limit} slots:"]
        lines.extend(
            f"#{hit.memory.id} [{hit.memory.type}] {hit.memory.content}" for hit in result.memories
        )
        if result.notes:
            lines.append(f"[{'; '.join(result.notes)}]")
        return ToolResult(ok=True, content="\n".join(lines))


def memory_tools(
    store: MemoryStore, workspace: str | Path | None = None
) -> tuple[RememberTool, RecallTool]:
    """Bind both tools to one store, for an agent run that has memory enabled."""
    root = Path(workspace) if workspace is not None else None
    return RememberTool(store, root), RecallTool(store, root)

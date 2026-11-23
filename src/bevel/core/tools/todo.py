"""The ``todo`` tool: the model's own task list for the session.

One tool that replaces the whole list. The model passes the complete list every
time; there are no ids, no ``action`` enum, and no merge semantics. That is
deliberate. An item-addressable interface asks a small model to remember an
identifier it invented three turns ago and to name an operation on it, which is
exactly where it starts inventing both. Rewriting a short list is something it
gets right, and "the list on disk is the list I just sent" is a property the
TUI and the CLI can then state without qualification.

Declared ``READ``, and the reason is written here because it will otherwise
look like a bug: a tool that writes to the database should not be ``READ``.
``WRITE_WORKSPACE`` would be wrong in the literal sense -- nothing here touches
a workspace file -- and the policy engine keys approvals on that tier, so every
progress update the model made would stop and ask in default mode. A tool whose
only effect is on the model's own scratch list, in a table scoped to its own
session, discarded with the session, is not a risk the user is being asked to
approve. Declaring it higher to be safe would be the inflated declaration the
tier ladder in ``base.py`` warns against, and would make the tool unusable.

The tool takes the store from :attr:`ToolContext.todos`. The registry holds a
store-less instance so the tool list is stable whether or not a session store
is attached; a call without one fails loudly rather than returning an empty
list, which is the difference between "no todos yet" and "todos are broken and
nobody noticed".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from bevel.core.session.store import Todo, TodoStatus
from bevel.core.tools.base import RiskTier, ToolContext, ToolResult
from bevel.core.tools.types import ToolSpec

__all__ = ["TodoTool"]

#: How long the list may get. The model rewrites the whole list on every call,
#: so the ceiling is set by what a small model reproduces faithfully, not by
#: anything about SQLite. Past this it starts dropping items it had written,
#: and a silently lost task is worse than a refused update.
MAX_ITEMS: Final[int] = 32

#: A todo is one line of intent, not a paragraph. Bounding it keeps the list
#: the model re-sends each turn small enough to rewrite exactly.
MAX_CONTENT: Final[int] = 200


class TodoTool:
    """Replace the session's task list with the one you are sending."""

    spec: Final[ToolSpec] = ToolSpec(
        name="todo",
        description=(
            "Record your plan for this session, or update it. Send the complete list "
            "every time: this replaces what is stored, it does not append. Mark a task "
            "done by sending the same list with that task's status set to 'done'. Send "
            "an empty list when the work is finished. Use it for anything more than "
            "trivial steps, so the user can watch the plan and you do not lose your "
            "place."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": (
                        "The whole plan, in the order it should be worked through. "
                        "This is the complete list: anything not in it is forgotten."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": (
                                    "The task, in a few words, phrased as the thing to do."
                                ),
                                "minLength": 1,
                                "maxLength": MAX_CONTENT,
                            },
                            "status": {
                                "type": "string",
                                "enum": [str(status) for status in TodoStatus],
                                "description": (
                                    "'pending' if you have not started it, 'in_progress' "
                                    "for the one task you are on now, 'done' once it is "
                                    "actually finished."
                                ),
                            },
                        },
                        "required": ["content", "status"],
                        "additionalProperties": False,
                    },
                    "minItems": 0,
                    "maxItems": MAX_ITEMS,
                },
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
    )
    risk_tier: Final[RiskTier] = RiskTier.READ
    is_parallel_safe: Final[bool] = False

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        store = ctx.todos
        if store is None:
            return ToolResult.failure("todo: no session store is configured for this session")
        if store.session_id is None:
            return ToolResult.failure("todo: no session is open yet, so there is nothing to store")

        todos = tuple(
            Todo(content=str(item["content"]).strip(), status=TodoStatus(str(item["status"])))
            for item in arguments["todos"]
        )
        try:
            store.replace(todos)
        except ValueError as exc:
            return ToolResult.failure(f"todo: {exc}")

        if not todos:
            return ToolResult(ok=True, content="todo list cleared.")
        done = sum(1 for todo in todos if todo.status is TodoStatus.DONE)
        marks = {TodoStatus.DONE: "x", TodoStatus.IN_PROGRESS: ">", TodoStatus.PENDING: " "}
        lines = [f"[{marks[todo.status]}] {todo.content}" for todo in todos]
        return ToolResult(
            ok=True,
            content=f"todo list saved, {done} of {len(todos)} done:\n" + "\n".join(lines),
        )

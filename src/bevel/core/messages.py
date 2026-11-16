"""Canonical, provider-agnostic message model.

The agent loop and the TUI speak these types; only ``providers/`` ever converts
them to and from a wire format. ``reasoning`` is a first-class field rather than
part of ``content`` because the endpoint this project targets is a reasoning
model that returns ``content: null`` alongside a populated ``reasoning`` --
folding the two together would either hide the chain of thought or leak it into
the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """A tool call as the model asked for it.

    ``arguments`` stays the raw JSON string the server sent. Parsing it is
    ``providers.repair``'s job, and keeping it unparsed here means a malformed
    call is still representable -- which is exactly the state the repair layer
    needs to receive.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Message:
    """One turn of a conversation.

    Invariants the providers rely on: a ``tool`` message carries ``tool_call_id``
    and nothing else; an assistant message that asked for tools has
    ``tool_calls`` non-empty.
    """

    role: Role
    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    @staticmethod
    def system(content: str) -> Message:
        return Message("system", content=content)

    @staticmethod
    def user(content: str) -> Message:
        return Message("user", content=content)

    @staticmethod
    def tool_result(tool_call_id: str, content: str) -> Message:
        return Message("tool", content=content, tool_call_id=tool_call_id)

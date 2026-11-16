"""Deterministic provider for tests, evals, and offline runs.

Replays a scripted transcript so the whole stack -- agent loop, repair, TUI,
evals -- runs with no GPU, no API key, and no network. This is not a stand-in
for the real provider in production; it is the reference implementation of
:class:`ModelProvider`, and the unit tests assert the real provider agrees with
it on identical wire payloads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from bevel.core.messages import Message, ToolCall
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import (
    AssistantTurn,
    Capabilities,
    ContentDelta,
    GenOptions,
    Health,
    ReasoningDelta,
    StreamEnd,
    StreamEvent,
    TokenCount,
    Usage,
    assemble,
)


@dataclass
class ScriptedTurn:
    """One canned reply.

    ``chunks`` are the content deltas to emit, in order; when omitted the whole
    ``content`` arrives as a single delta. Tool calls are delivered on the
    terminal event, matching how the real provider assembles them.
    """

    content: str = ""
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks: tuple[str, ...] = ()
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.chunks and self.content:
            object.__setattr__(self, "chunks", (self.content,))


def tool_call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    """Convenience for tests: build a :class:`ToolCall` from a dict."""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


@dataclass
class MockProvider:
    """A :class:`ModelProvider` that answers from a list.

    Requests are recorded so a test can assert on exactly what the model was
    asked -- which is how the repair layer is tested without a server. Turns
    cycle, so a short script can drive a long conversation.
    """

    turns: list[ScriptedTurn] = field(default_factory=list)
    name: str = "mock"
    model: str = "mock-model"
    capabilities: Capabilities = field(default_factory=Capabilities)
    health_ok: bool = True
    health_detail: str = "mock provider"
    token_count: int | None = None
    delay: float = 0.0
    requests: list[dict[str, object]] = field(default_factory=list)
    _index: int = 0

    def _next_turn(self) -> ScriptedTurn:
        if not self.turns:
            return ScriptedTurn(content="")
        turn = self.turns[self._index % len(self.turns)]
        self._index += 1
        return turn

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream(messages, tools, options)

    async def _stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        options: GenOptions | None,
    ) -> AsyncIterator[StreamEvent]:
        self.requests.append(
            {
                "messages": list(messages),
                "tools": [t.name for t in tools] if tools else [],
                "options": options,
            }
        )
        turn = self._next_turn()
        if self.delay:
            await asyncio.sleep(self.delay)
        if turn.reasoning:
            yield ReasoningDelta(turn.reasoning)
        for chunk in turn.chunks:
            yield ContentDelta(chunk)

        completion = turn.completion_tokens or sum(len(c) for c in turn.chunks)
        yield StreamEnd(
            finish_reason=turn.finish_reason,  # type: ignore[arg-type]
            usage=Usage(
                prompt_tokens=turn.prompt_tokens,
                completion_tokens=completion,
                total_tokens=turn.prompt_tokens + completion,
            ),
            tool_calls=turn.tool_calls,
            model=turn.model or self.model,
        )

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AssistantTurn:
        return await assemble(self.stream(messages, tools, options))

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:  # noqa: ARG002
        # ``messages`` is part of the ModelProvider signature; the mock answers
        # from configuration rather than by inspecting them.
        if self.token_count is None:
            return TokenCount(None, source="mock has no tokenizer")
        return TokenCount(self.token_count, source="mock")

    async def health(self) -> Health:
        return Health(self.health_ok, self.health_detail, models=(self.model,))

    async def aclose(self) -> None:
        return None

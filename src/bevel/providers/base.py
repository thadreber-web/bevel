"""The provider seam.

Everything above this file -- the agent loop, the TUI, the budget guard --
speaks :class:`ModelProvider` and never learns that HTTP exists. Capabilities
are *reported* by the implementation, which obtains them by probing, and
:class:`~bevel.providers.registry.Registry` caches them; nothing here assumes a
server is OpenAI-compatible because it is called OpenAI-compatible.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from bevel.core.messages import Message, ToolCall
from bevel.core.tools.types import ToolSpec

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "abort"]
"""``abort`` means the stream ended without the server ever sending a finish
reason. That is a real case on this endpoint's gateway, not a hypothetical."""


class ProviderError(Exception):
    """Base class for everything a provider raises."""


class TransportError(ProviderError):
    """The request never got a usable answer: DNS, TLS, timeout, connection.

    Not the model's fault. Repair must not re-ask the model over this.
    """


class APIError(ProviderError):
    """The server answered with an error body.

    ``retryable`` is decided from the status code, not from the message text.
    """

    def __init__(self, status: int, message: str, *, code: str | None = None) -> None:
        super().__init__(
            f"HTTP {status}: {message}" if code is None else f"HTTP {status} [{code}]: {message}"
        )
        self.status = status
        self.code = code
        self.retryable = status == 429 or status >= 500


class MalformedResponseError(ProviderError):
    """A response we could read but not interpret.

    Raised for a body that is not JSON, or a shape that contradicts the
    documented contract. Callers get the raw text in ``body`` so the log can
    show what actually arrived.
    """

    def __init__(self, message: str, body: str = "") -> None:
        super().__init__(message)
        self.body = body


class StreamAbortedError(ProviderError):
    """The stream stopped before it was allowed to finish."""


@dataclass(frozen=True)
class Capabilities:
    """What a specific endpoint was observed to support.

    Every flag is the result of a probe, and the defaults are the pessimistic
    ones: an endpoint that has not demonstrated a feature does not have it.
    ``max_model_len`` is ``None`` when the server did not report it -- the
    budget guard then refuses to predict rather than inventing a window.
    """

    supports_streaming: bool = True
    supports_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_token_api: bool = False
    supports_embeddings: bool = False
    max_model_len: int | None = None
    reasoning: bool = False

    @property
    def summary(self) -> str:
        """Short human-readable form for ``bevel doctor`` and the TUI footer."""
        on = [
            name[len("supports_") :]
            for name in (
                "supports_streaming",
                "supports_tools",
                "supports_parallel_tool_calls",
                "supports_structured_output",
                "supports_token_api",
                "supports_embeddings",
            )
            if getattr(self, name)
        ]
        if self.reasoning:
            on.append("reasoning")
        window = "?" if self.max_model_len is None else str(self.max_model_len)
        return f"ctx={window} " + ",".join(on)


@dataclass(frozen=True)
class Usage:
    """Token counts *reported by the server* for a completed request.

    These are the only trustworthy numbers in the system. They are retrospective:
    they tell you what a request cost, never what the next one will.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class TokenCount:
    """The result of a token count request.

    ``tokens`` is ``None`` when the endpoint cannot count. There is deliberately
    no way to construct one of these from a character-length guess: a number
    that looks authoritative is worse than an honest blank.
    """

    tokens: int | None
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return self.tokens is not None


@dataclass(frozen=True)
class GenOptions:
    """Per-request generation options. ``None`` means "do not send the field"."""

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    json_mode: bool = False


@dataclass(frozen=True)
class ReasoningDelta:
    text: str


@dataclass(frozen=True)
class ContentDelta:
    text: str


@dataclass(frozen=True)
class StreamEnd:
    """Terminal event. Always the last event of a well-formed stream."""

    finish_reason: FinishReason
    usage: Usage | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    model: str | None = None


StreamEvent = ReasoningDelta | ContentDelta | StreamEnd


@dataclass(frozen=True)
class AssistantTurn:
    """A fully-assembled assistant reply."""

    message: Message
    finish_reason: FinishReason
    usage: Usage | None = None
    model: str | None = None


@dataclass(frozen=True)
class Health:
    """Result of a liveness probe.

    ``detail`` says what was actually tried, so ``bevel doctor`` can distinguish
    "endpoint is down" from "endpoint has no /health route and /v1/models worked".
    """

    ok: bool
    detail: str
    latency_ms: float | None = None
    capabilities: Capabilities | None = None
    models: tuple[str, ...] = ()


@runtime_checkable
class ModelProvider(Protocol):
    """The only interface the rest of bevel has to a model server."""

    name: str
    model: str
    capabilities: Capabilities

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield deltas as they arrive, ending with exactly one :class:`StreamEnd`."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        options: GenOptions | None = None,
    ) -> AssistantTurn:
        """Collect :meth:`stream` into a single turn."""
        ...

    async def count_tokens(self, messages: Sequence[Message]) -> TokenCount:
        """Return an exact count, or ``TokenCount(None)``. Never an estimate."""
        ...

    async def embed(self, text: str) -> tuple[float, ...]:
        """Embedding for ``text`` from ``/embeddings``.

        Raises :class:`ProviderError` when the endpoint does not serve the
        route. There is no fallback vector: a zero or a guess would turn every
        cosine into a confident wrong answer, and recall would look like it
        worked. Callers check ``capabilities.supports_embeddings`` first and
        degrade to lexical recall when the answer is no.
        """
        ...

    async def health(self) -> Health:
        """Cheap liveness probe. Must not generate tokens."""
        ...

    async def aclose(self) -> None:
        """Release network resources. Safe to call twice."""
        ...


@dataclass
class _Accumulator:
    """Mutable state used by :func:`assemble`."""

    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    finish_reason: FinishReason = "abort"
    usage: Usage | None = None
    model: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


async def assemble(events: AsyncIterator[StreamEvent]) -> AssistantTurn:
    """Fold a stream into an :class:`AssistantTurn`.

    Shared by every provider's :meth:`ModelProvider.complete` so the streaming
    and non-streaming paths cannot drift apart -- there is one assembly rule
    and it lives here.
    """
    acc = _Accumulator()
    async for event in events:
        if isinstance(event, ContentDelta):
            acc.content.append(event.text)
        elif isinstance(event, ReasoningDelta):
            acc.reasoning.append(event.text)
        else:
            acc.finish_reason = event.finish_reason
            acc.usage = event.usage
            acc.tool_calls = event.tool_calls
            acc.model = event.model

    reasoning = "".join(acc.reasoning)
    return AssistantTurn(
        message=Message(
            "assistant",
            content="".join(acc.content),
            reasoning=reasoning or None,
            tool_calls=acc.tool_calls,
        ),
        finish_reason=acc.finish_reason,
        usage=acc.usage,
        model=acc.model,
    )

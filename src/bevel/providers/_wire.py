"""Wire-format decoding for OpenAI-compatible chat completions.

Kept apart from the HTTP layer on purpose: every surprising shape this endpoint
family produces is handled here, and each can be tested with a string instead of
a GPU. The behaviours encoded below were measured against a live vLLM behind a
gateway, not read off the OpenAI documentation.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

from bevel.core.messages import ToolCall
from bevel.providers.base import (
    APIError,
    ContentDelta,
    FinishReason,
    MalformedResponseError,
    ReasoningDelta,
    StreamEnd,
    StreamEvent,
    Usage,
)

DONE_SENTINEL = "[DONE]"

USAGE_GRACE_CHUNKS = 8
"""How many payloads to keep reading after a finish reason.

vLLM puts the usage chunk *after* the chunk that carries ``finish_reason``,
so stopping the moment we see a finish reason silently throws away the token
count the budget guard depends on. We keep reading for a small, bounded
number of extra payloads instead: enough for the usage chunk and ``[DONE]``,
but not enough to hang forever behind a gateway that holds the connection
open.
"""

_VALID_FINISH_REASONS: frozenset[str] = frozenset(
    {"stop", "length", "tool_calls", "content_filter"}
)
"""Anything else -- including vLLM's raw ``stop_reason`` strings -- becomes
``abort``, so the rest of bevel never sees a finish reason it has no policy
for."""


def normalise_finish_reason(raw: Any) -> FinishReason:
    """Map a server-supplied finish reason onto our vocabulary.

    Membership in :data:`_VALID_FINISH_REASONS` is what makes the ``cast``
    sound; anything unrecognised degrades to ``abort`` rather than leaking a
    string the rest of bevel has no policy for.
    """
    if isinstance(raw, str) and raw in _VALID_FINISH_REASONS:
        return cast("FinishReason", raw)
    return "abort"


@dataclass
class _CallState:
    """Accumulation slot for one in-flight tool call, keyed by wire ``index``.

    Keyed by ``index`` and not by ``id`` because only the first fragment of a
    given call carries the id; later fragments carry ``{"index": n, "function":
    {"arguments": "..."}}`` and nothing else.
    """

    id: str = ""
    name: str = ""
    arguments: list[str] = field(default_factory=list)

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments="".join(self.arguments))


def _require_dict(value: Any, what: str, body: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedResponseError(f"expected {what} to be an object", body=body)
    return value


def parse_usage(raw: Any) -> Usage | None:
    """Build a :class:`Usage` from a ``usage`` object, or ``None`` if absent.

    ``total_tokens`` is taken from the server when present and derived
    otherwise; it is arithmetic on server-reported numbers, which is not an
    estimate.
    """
    if raw is None:
        return None
    usage = _require_dict(raw, "usage", "")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    details = usage.get("completion_tokens_details")
    reasoning = 0
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, int):
            reasoning = value
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        total = prompt + completion
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=total,
    )


def parse_error_body(status: int, body: str) -> tuple[str, str | None]:
    """Extract ``(message, code)`` from an error body.

    Handles the two shapes this endpoint actually produces -- the FastAPI
    ``{"detail": ...}`` form and the OpenAI ``{"error": {...}}`` form -- and
    falls back to truncated raw text so a log line is never empty.
    """
    text = body.strip()
    if not text:
        return (f"empty body with status {status}", None)

    message: str | None = None
    code: str | None = None
    parsed: Any = None
    with contextlib.suppress(json.JSONDecodeError):
        parsed = json.loads(text)

    if isinstance(parsed, dict):
        detail = parsed.get("detail")
        if isinstance(detail, str):
            message = detail
        elif isinstance(detail, list) and detail:
            message = _fastapi_detail(detail)
        else:
            error = parsed.get("error")
            if isinstance(error, dict):
                raw_message = error.get("message")
                if isinstance(raw_message, str):
                    message = raw_message
                raw_code = error.get("code")
                code = str(raw_code) if raw_code is not None else None
            elif isinstance(error, str):
                message = error

    return (message if message is not None else _clip(text), code)


def _clip(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _fastapi_detail(detail: list[Any]) -> str | None:
    """Turn FastAPI's validation-error list into something actionable.

    ``{"detail": [{"loc": ["body", "model"], "msg": "field required"}]}``
    becomes ``"body.model: field required"``. The ``loc`` path is the whole
    point -- "field required" on its own tells you nothing about which of a
    dozen fields the server rejected.
    """
    first = detail[0]
    if not isinstance(first, dict):
        return None
    msg = first.get("msg")
    if not isinstance(msg, str):
        return None

    loc = first.get("loc")
    path = ".".join(str(part) for part in loc) if isinstance(loc, (list, tuple)) else ""
    message = f"{path}: {msg}" if path else msg
    if len(detail) > 1:
        message = f"{message} (+{len(detail) - 1} more)"
    return message


class ChatStreamDecoder:
    """Turns SSE ``data:`` payloads into :data:`StreamEvent` values.

    Stateful across chunks: tool-call fragments arrive split over many events
    and have to be reassembled per ``index``.

    Two shapes drive the design, both observed live:

    * the usage chunk carries ``"choices": []``, so ``choices[0]`` must never
      be indexed unconditionally;
    * reasoning arrives as ``delta.reasoning`` while ``delta.content`` is still
      absent, so the two are tracked as separate channels.
    """

    def __init__(self) -> None:
        self._calls: dict[int, _CallState] = {}
        self._order: list[int] = []
        self._finish: FinishReason | None = None
        self._usage: Usage | None = None
        self._model: str | None = None
        self._done = False
        self._chunks_after_finish = 0

    def feed(self, payload: str) -> list[StreamEvent]:
        """Decode one SSE data payload. Empty list for keepalives and for the
        usage-only chunk, which carries no visible text."""
        text = payload.strip()
        if not text:
            return []
        if self._finish is not None:
            self._chunks_after_finish += 1
        if text == DONE_SENTINEL:
            self._done = True
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedResponseError(f"stream chunk is not JSON: {exc}", body=text) from exc

        obj = _require_dict(obj, "chunk", text)
        if isinstance(obj.get("error"), (dict, str)):
            # An in-band error is the same failure as an out-of-band one -- the
            # server told us it could not do the job -- so it raises the same
            # exception and the caller has one handling path, not two.
            message, code = parse_error_body(200, text)
            raise APIError(200, message, code=code)
        if code := obj.get("error_code"):
            raise MalformedResponseError(f"stream error code {code}", body=text)

        if isinstance(obj.get("model"), str):
            self._model = str(obj["model"])
        self._usage = parse_usage(obj.get("usage")) or self._usage

        choices = obj.get("choices")
        if choices is None or choices == []:
            return []
        if not isinstance(choices, list) or not all(isinstance(c, dict) for c in choices):
            raise MalformedResponseError("chunk has a malformed choices array", body=text)

        events: list[StreamEvent] = []
        for choice in choices:
            events.extend(self._feed_choice(choice, text))
        return events

    def _feed_choice(self, choice: dict[str, Any], body: str) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        raw_reason = choice.get("finish_reason")
        if raw_reason is not None:
            self._finish = normalise_finish_reason(raw_reason)

        delta = choice.get("delta")
        if delta is None:
            return events
        delta = _require_dict(delta, "delta", body)

        if isinstance(delta.get("reasoning"), str):
            events.append(ReasoningDelta(delta["reasoning"]))
        if isinstance(delta.get("reasoning_content"), str):
            # llama.cpp and some vLLM builds use this key instead.
            events.append(ReasoningDelta(delta["reasoning_content"]))
        if isinstance(delta.get("content"), str):
            events.append(ContentDelta(delta["content"]))

        calls = delta.get("tool_calls")
        if isinstance(calls, list):
            for entry in calls:
                if isinstance(entry, dict):
                    self._feed_tool_call(entry)
        return events

    def _feed_tool_call(self, entry: dict[str, Any]) -> None:
        index = entry.get("index")
        if not isinstance(index, int):
            index = len(self._order)
        state = self._calls.get(index)
        if state is None:
            state = _CallState()
            self._calls[index] = state
            self._order.append(index)

        if isinstance(entry.get("id"), str):
            state.id = entry["id"]
        function = entry.get("function")
        if isinstance(function, dict):
            if isinstance(function.get("name"), str):
                state.name += function["name"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                state.arguments.append(arguments)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(self._calls[i].to_tool_call() for i in self._order)

    @property
    def finished(self) -> bool:
        """True once there is nothing more worth reading from the stream.

        Not simply "a finish reason arrived": the usage chunk follows it, so
        we hold on for a bounded number of extra payloads. ``[DONE]`` ends it
        immediately, and the counter stops us hanging on a gateway that keeps
        the connection open.
        """
        if self._done:
            return True
        return self._finish is not None and self._chunks_after_finish >= USAGE_GRACE_CHUNKS

    def finalize(self) -> StreamEnd:
        """Produce the terminal event.

        A stream that ends without either ``[DONE]`` or a finish reason is an
        ``abort`` -- reported as such rather than being silently treated as a
        clean stop.
        """
        return StreamEnd(
            finish_reason=self._finish if self._finish is not None else "abort",
            usage=self._usage,
            tool_calls=self.tool_calls,
            model=self._model,
        )


def iter_sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Yield the payload of each ``data:`` line, joining multi-line events.

    Accepts both ``data: x`` and ``data:x``. Comment lines (``:``) and
    non-``data`` fields are ignored, which is what the SSE spec asks of a
    client.
    """
    buffered: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if buffered:
                yield "\n".join(buffered)
                buffered = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buffered.append(line[len("data:") :].lstrip())
    if buffered:
        yield "\n".join(buffered)


def parse_completion_response(
    body: str,
) -> tuple[str, str | None, FinishReason, Usage | None, str | None]:
    """Decode a non-streaming ``/chat/completions`` body.

    Returns ``(content, reasoning, finish_reason, usage, model)``. ``content``
    is ``""`` when the model returned only reasoning, which is the normal case
    for a reasoning model cut off mid-thought.
    """
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"response is not JSON: {exc}", body=body) from exc

    obj = _require_dict(obj, "response", body)
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MalformedResponseError("response has no choices", body=body)
    choice = _require_dict(choices[0], "choices[0]", body)

    message = choice.get("message")
    if message is None:
        message = {}
    message = _require_dict(message, "message", body)

    content = message.get("content")
    text = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning_content")
    reasoning_value = reasoning if isinstance(reasoning, str) else None

    raw_reason = choice.get("finish_reason")
    finish: FinishReason = normalise_finish_reason(raw_reason)
    model = obj.get("model")
    return (
        text,
        reasoning_value,
        finish,
        parse_usage(obj.get("usage")),
        model if isinstance(model, str) else None,
    )


def parse_tool_calls_from_message(body: str) -> tuple[ToolCall, ...]:
    """Read ``message.tool_calls`` from a non-streaming body.

    The non-streaming shape carries each call complete rather than in
    fragments, so this reads it directly; ``arguments`` is still handed on
    unparsed for the repair layer to handle.
    """
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"response is not JSON: {exc}", body=body) from exc

    obj = _require_dict(obj, "response", body)
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return ()
    choice = _require_dict(choices[0], "choices[0]", body)
    message = _require_dict(choice.get("message") or {}, "message", body)
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return ()

    result: list[ToolCall] = []
    for position, entry in enumerate(calls):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        call_id = entry.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        result.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) else f"call_{position}",
                name=name if isinstance(name, str) else "",
                arguments=arguments if isinstance(arguments, str) else "",
            )
        )
    return tuple(result)

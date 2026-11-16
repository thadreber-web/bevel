"""Tests against a real endpoint.

Two kinds of test live here and they behave differently on purpose.

*The invariants* hold for any server that claims to speak this API: a stream ends
with exactly one :class:`StreamEnd`, usage is server-reported or absent, and
``count_tokens`` never invents a number. These should pass against vLLM, llama.cpp,
or a mock pretending to be either.

*The pinned expectations* encode what was actually measured on the author's vLLM
server (see ``bevel doctor``). They are allowed to fail -- a failure means the
endpoint changed underneath bevel, which is precisely the moment the capability
cache must be invalidated rather than trusted.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from bevel.core.messages import Message
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import (
    ContentDelta,
    GenOptions,
    ReasoningDelta,
    StreamEnd,
    StreamEvent,
    Usage,
)
from bevel.providers.openai_compat import OpenAICompatProvider

# The endpoint fixture is module-scoped (one probe for the whole module), so these
# tests must share its event loop scope rather than the project default of "function".
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

WEATHER = ToolSpec(
    name="get_weather",
    description="Return the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)


async def _collect(
    provider: OpenAICompatProvider,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None = None,
    options: GenOptions | None = None,
) -> list[StreamEvent]:
    return [event async for event in provider.stream(messages, tools, options)]


# --------------------------------------------------------------------------- invariants


async def test_health_lists_the_configured_model(endpoint: OpenAICompatProvider) -> None:
    health = await endpoint.health()
    assert health.ok, health.detail
    assert health.models, "a healthy endpoint must list at least one model"
    assert endpoint.model in health.models, (
        f"{endpoint.model!r} is not among {health.models}; the profile names a model "
        "this server does not serve"
    )
    assert health.latency_ms is not None, "a healthy endpoint must report its latency"
    assert health.latency_ms >= 0.0


async def test_probe_reports_capabilities_without_raising(endpoint: OpenAICompatProvider) -> None:
    caps = await endpoint.probe()
    assert caps == endpoint.capabilities, "probe() must publish what it measured"
    # supports_streaming is not probed -- it is a precondition of using this class.
    assert caps.supports_streaming is True


async def test_count_tokens_never_estimates(endpoint: OpenAICompatProvider) -> None:
    count = await endpoint.count_tokens([Message.user("Count these tokens please.")])
    if count.tokens is None:
        # The honest answer. It must say *why*, so the TUI can explain itself.
        assert count.source != "unknown", "an unknown count needs a source explaining it"
    else:
        assert count.tokens > 0
        assert endpoint.capabilities.supports_token_api is True, (
            "a numeric count arrived from an endpoint that does not advertise /tokenize"
        )


async def test_stream_ends_with_exactly_one_stream_end(endpoint: OpenAICompatProvider) -> None:
    events = await _collect(endpoint, [Message.user("Reply with the single word: ready")])
    ends = [event for event in events if isinstance(event, StreamEnd)]
    assert len(ends) == 1, f"expected exactly one StreamEnd, got {len(ends)}: {events!r}"
    assert events[-1] is ends[0], "StreamEnd must be the last event of the stream"
    assert ends[0].finish_reason in {"stop", "length", "tool_calls", "content_filter", "abort"}


async def test_usage_is_reported_when_the_server_is_asked_for_it(
    endpoint: OpenAICompatProvider,
) -> None:
    """The budget guard runs on these numbers, so they must actually arrive.

    vLLM sends usage in a final chunk whose ``choices`` list is empty, *after* the
    chunk carrying ``finish_reason``. A decoder that treats the finish reason as
    terminal discards it and the guard silently sees zero.
    """
    events = await _collect(endpoint, [Message.user("Say the word: ready")])
    end = next(event for event in events if isinstance(event, StreamEnd))
    assert isinstance(end.usage, Usage), f"no usage in {end!r}"
    assert end.usage is not None
    assert end.usage.prompt_tokens > 0, "server reported zero prompt tokens"
    assert end.usage.total_tokens >= end.usage.prompt_tokens


async def test_content_deltas_concatenate_to_the_reply(
    endpoint: OpenAICompatProvider,
) -> None:
    messages = [Message.user("Reply with exactly: alpha beta gamma")]
    events = await _collect(endpoint, messages, options=GenOptions(temperature=0.0))
    streamed = "".join(event.text for event in events if isinstance(event, ContentDelta))
    assert streamed.strip(), "streaming produced no content at all"

    # The other half goes over a different server-side code path, so it is worth
    # asking the same question of it rather than trusting the stream alone.
    turn = await endpoint.complete_non_streaming(messages, options=GenOptions(temperature=0.0))
    assert turn.message.content.strip(), "non-streaming produced no content at all"
    assert turn.finish_reason != "abort", f"the reply was aborted: {turn!r}"


async def test_reasoning_arrives_on_its_own_channel(endpoint: OpenAICompatProvider) -> None:
    """Reasoning must never be folded into content.

    On this reasoning model the non-streaming reply has ``content: null`` and the
    chain of thought in ``reasoning``; streaming emits ``delta.reasoning`` chunks
    before ``delta.content`` begins. Merging the two would put the model's private
    deliberation into the transcript the user reads.
    """
    if not endpoint.capabilities.reasoning:
        pytest.skip("endpoint did not demonstrate a reasoning channel")

    turn = await endpoint.complete([Message.user("Think about 7 * 8, then answer.")])
    assert turn.message.reasoning, "capabilities claim reasoning but the reply has none"
    assert turn.message.reasoning not in turn.message.content

    events = await _collect(endpoint, [Message.user("Think about 11 * 12, then answer.")])
    reasoning = "".join(event.text for event in events if isinstance(event, ReasoningDelta))
    content = "".join(event.text for event in events if isinstance(event, ContentDelta))
    assert reasoning, "capabilities claim reasoning but the stream carried no reasoning deltas"
    assert reasoning not in content, "reasoning text leaked into the content channel"


async def test_tool_calls_survive_the_stream(endpoint: OpenAICompatProvider) -> None:
    if not endpoint.capabilities.supports_tools:
        pytest.skip("endpoint did not demonstrate tool calling")

    events = await _collect(
        endpoint,
        [Message.user("What is the weather in Paris? Use the get_weather tool.")],
        tools=[WEATHER],
        options=GenOptions(temperature=0.0),
    )
    end = next(event for event in events if isinstance(event, StreamEnd))

    assert end.tool_calls, f"model returned no tool calls: {end!r}"
    call = end.tool_calls[0]
    assert call.name == WEATHER.name
    assert call.id, "a tool call without an id cannot be answered"
    # Fragments arrive split across chunks and must be concatenated per index.
    arguments = json.loads(call.arguments)
    assert isinstance(arguments, dict), f"arguments were not a JSON object: {call.arguments!r}"
    assert "city" in arguments, f"required property missing from {arguments!r}"


async def test_parallel_tool_calls_are_separate_calls(endpoint: OpenAICompatProvider) -> None:
    if not endpoint.capabilities.supports_parallel_tool_calls:
        pytest.skip("endpoint did not demonstrate parallel tool calls")

    turn = await endpoint.complete(
        [Message.user("What is the weather in Paris and in Berlin? Call the tool twice.")],
        tools=[WEATHER],
        options=GenOptions(temperature=0.0),
    )
    assert len(turn.message.tool_calls) >= 2, (
        f"capabilities claim parallel calls but the model returned "
        f"{len(turn.message.tool_calls)}: {turn.message.tool_calls!r}"
    )
    ids = [call.id for call in turn.message.tool_calls]
    assert len(set(ids)) == len(ids), f"parallel calls share ids: {ids!r}"


# --------------------------------------------------------------- pinned to the author's


async def test_measured_capabilities_match_the_probed_endpoint(
    endpoint: OpenAICompatProvider,
) -> None:
    """Locks the facts measured on the author's vLLM server.

    This test failing is information, not a bug: it means the endpoint changed --
    a route appeared, the context window moved, or a gateway started proxying
    ``/tokenize``. Re-run ``bevel doctor`` and update the cache, do not delete the
    assertion.
    """
    caps = endpoint.capabilities
    assert caps.supports_tools is True
    assert caps.supports_parallel_tool_calls is True
    assert caps.max_model_len == 131072
    assert caps.reasoning is True
    # Absent on this server: the gateway does not proxy them. If these flip to
    # True, bevel should start using them -- and this test is the nudge.
    assert caps.supports_token_api is False
    assert caps.supports_embeddings is False

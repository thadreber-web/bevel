"""Provider tests against an in-process transport -- no network, no GPU.

``httpx.MockTransport`` gives us a real HTTP stack (so header, status and
body handling are genuinely exercised) while letting each test dictate the
server's exact answer, including the answers the real endpoint gives that
the OpenAI documentation does not mention.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from bevel.core.messages import Message, ToolCall
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import (
    APIError,
    Capabilities,
    ContentDelta,
    GenOptions,
    MalformedResponseError,
    ReasoningDelta,
    StreamEnd,
    TokenCount,
    TransportError,
)
from bevel.providers.openai_compat import OpenAICompatProvider, build_request

MODEL = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
BASE = "https://example.test/v1"


def make_provider(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAICompatProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(
        name="test",
        base_url=BASE,
        model=MODEL,
        api_key="secret",
        client=client,
        **kwargs,
    )


async def collect(provider: OpenAICompatProvider, messages: Sequence[Message]) -> list[Any]:
    return [event async for event in provider.stream(messages)]


# -- build_request ---------------------------------------------------------


class TestBuildRequest:
    def test_minimal_body(self) -> None:
        body = build_request("m", [Message.user("hi")], None, None, stream=False)
        assert body == {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        }

    def test_unset_options_are_omitted_not_nulled(self) -> None:
        """``temperature: null`` is not the same as omitting it on vLLM."""
        body = build_request("m", [Message.user("hi")], None, GenOptions(), stream=False)
        assert "temperature" not in body
        assert "top_p" not in body
        assert "max_tokens" not in body
        assert "response_format" not in body

    def test_set_options_are_sent(self) -> None:
        body = build_request(
            "m",
            [Message.user("hi")],
            None,
            GenOptions(temperature=0.2, top_p=0.9, max_tokens=64),
            stream=False,
        )
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.9
        assert body["max_tokens"] == 64

    def test_json_mode_uses_response_format(self) -> None:
        body = build_request(
            "m", [Message.user("hi")], None, GenOptions(json_mode=True), stream=False
        )
        assert body["response_format"] == {"type": "json_object"}

    def test_stream_options_request_usage(self) -> None:
        """Without this the stream carries no usage and the budget guard is blind."""
        body = build_request("m", [Message.user("hi")], None, None, stream=True)
        assert body["stream_options"] == {"include_usage": True}

    def test_no_stream_options_when_not_streaming(self) -> None:
        body = build_request("m", [Message.user("hi")], None, None, stream=False)
        assert "stream_options" not in body

    def test_tools_are_wrapped_for_the_wire(self) -> None:
        spec = ToolSpec(name="read", description="Read a file.", parameters={"type": "object"})
        body = build_request("m", [Message.user("hi")], [spec], None, stream=False)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file.",
                    "parameters": {"type": "object"},
                },
            }
        ]

    def test_empty_tool_list_is_omitted(self) -> None:
        body = build_request("m", [Message.user("hi")], [], None, stream=False)
        assert "tools" not in body


class TestMessageSerialization:
    def body_for(self, message: Message) -> dict[str, Any]:
        return build_request("m", [message], None, None, stream=False)["messages"][0]

    def test_tool_result_carries_tool_call_id(self) -> None:
        payload = self.body_for(Message.tool_result("call_1", "the output"))
        assert payload["role"] == "tool"
        assert payload["tool_call_id"] == "call_1"
        assert payload["content"] == "the output"

    def test_assistant_tool_calls_send_empty_content(self) -> None:
        """A null content alongside tool_calls is rejected by some servers."""
        message = Message("assistant", content=None, tool_calls=(ToolCall("c1", "read", "{}"),))
        payload = self.body_for(message)
        assert payload["content"] == ""
        assert payload["tool_calls"] == [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ]

    def test_assistant_with_both_keeps_content(self) -> None:
        message = Message(
            "assistant", content="thinking", tool_calls=(ToolCall("c1", "read", "{}"),)
        )
        assert self.body_for(message)["content"] == "thinking"

    def test_arguments_are_passed_through_unparsed(self) -> None:
        broken = '{"path": "x",}'
        message = Message("assistant", tool_calls=(ToolCall("c1", "read", broken),))
        assert self.body_for(message)["tool_calls"][0]["function"]["arguments"] == broken


# -- streaming -------------------------------------------------------------


def sse(chunks: list[dict[str, Any]], *, done: bool = True) -> httpx.Response:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    if done:
        lines.append("data: [DONE]\n\n")
    return httpx.Response(200, text="".join(lines), headers={"Content-Type": "text/event-stream"})


def content_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


class TestStream:
    async def test_yields_content_deltas_then_end(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return sse(
                [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                    content_chunk("he"),
                    content_chunk("llo"),
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            )

        events = await collect(make_provider(handler), [Message.user("hi")])
        assert events == [
            ContentDelta("he"),
            ContentDelta("llo"),
            StreamEnd("stop"),
        ]

    async def test_reasoning_arrives_on_its_own_channel(self) -> None:
        """The target model is a reasoning model; this is its normal shape."""

        def handler(request: httpx.Request) -> httpx.Response:
            return sse(
                [
                    {"choices": [{"index": 0, "delta": {"reasoning": "let me think"}}]},
                    {"choices": [{"index": 0, "delta": {"reasoning": " further"}}]},
                    content_chunk("42"),
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            )

        events = await collect(make_provider(handler), [Message.user("17*23?")])
        assert events == [
            ReasoningDelta("let me think"),
            ReasoningDelta(" further"),
            ContentDelta("42"),
            StreamEnd("stop"),
        ]

    async def test_usage_survives_the_empty_choices_chunk(self) -> None:
        """vLLM's final chunk has ``choices: []``; indexing it would IndexError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return sse(
                [
                    content_chunk("hi"),
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 5,
                            "total_tokens": 16,
                            "completion_tokens_details": {"reasoning_tokens": 3},
                        },
                    },
                ]
            )

        events = await collect(make_provider(handler), [Message.user("hi")])
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.usage is not None
        assert end.usage.prompt_tokens == 11
        assert end.usage.reasoning_tokens == 3

    async def test_stream_ends_with_abort_when_server_never_says_stop(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return sse([content_chunk("partial")])

        events = await collect(make_provider(handler), [Message.user("hi")])
        assert isinstance(events[-1], StreamEnd)
        assert events[-1].finish_reason == "abort"

    async def test_tool_call_fragments_are_reassembled(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return sse(
                [
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "read", "arguments": ""},
                                        }
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '{"path":'}}
                                    ]
                                },
                            }
                        ]
                    },
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"a.py"}'}}
                                    ]
                                },
                            }
                        ]
                    },
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                ]
            )

        events = await collect(make_provider(handler), [Message.user("read it")])
        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.tool_calls == (ToolCall("call_1", "read", '{"path":"a.py"}'),)

    async def test_request_body_and_headers(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return sse(
                [
                    content_chunk("ok"),
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            )

        provider = make_provider(handler)
        await collect(provider, [Message.user("hi")])
        request = seen[0]
        assert request.url == f"{BASE}/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content)["stream"] is True

    async def test_error_inside_a_200_stream_raises_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return sse([{"error": {"message": "model not loaded", "code": 400}}])

        with pytest.raises(APIError) as excinfo:
            await collect(make_provider(handler), [Message.user("hi")])
        # status stays 200 because that is what the transport actually
        # reported; the server's own code travels in `code`. One exception
        # type either way, so callers do not branch on where the error came
        # from.
        assert excinfo.value.status == 200
        assert excinfo.value.code == "400"
        assert "model not loaded" in str(excinfo.value)


# -- errors ----------------------------------------------------------------


class TestErrorHandling:
    async def test_non_2xx_raises_api_error_with_server_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "max_tokens too large", "code": "invalid_request"}},
            )

        with pytest.raises(APIError) as excinfo:
            await collect(make_provider(handler), [Message.user("hi")])
        assert excinfo.value.status == 400
        assert "max_tokens too large" in str(excinfo.value)

    async def test_fastapi_detail_shape_is_understood(self) -> None:
        """The gateway in front of vLLM answers with FastAPI shapes, not OpenAI ones."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                422, json={"detail": [{"loc": ["body", "model"], "msg": "missing"}]}
            )

        with pytest.raises(APIError) as excinfo:
            await collect(make_provider(handler), [Message.user("hi")])
        assert "model" in str(excinfo.value)

    async def test_timeout_becomes_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(TransportError) as excinfo:
            await collect(make_provider(handler), [Message.user("hi")])
        assert "timed out" in str(excinfo.value)

    async def test_connect_failure_becomes_transport_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        with pytest.raises(TransportError) as excinfo:
            await collect(make_provider(handler), [Message.user("hi")])
        assert "could not reach" in str(excinfo.value)

    async def test_non_streaming_path_raises_too(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="No available route")

        with pytest.raises(APIError) as excinfo:
            await make_provider(handler).complete_non_streaming([Message.user("hi")])
        assert excinfo.value.status == 503
        assert "No available route" in str(excinfo.value)


# -- non-streaming completion ----------------------------------------------


class TestCompleteNonStreaming:
    async def test_reasoning_model_returns_empty_content_with_reasoning(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": None, "reasoning": "hm"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
                },
            )

        turn = await make_provider(handler).complete_non_streaming([Message.user("hi")])
        # parse_completion_response normalises a null content to "" so callers
        # never have to branch on None; the reasoning is the real payload.
        assert turn.message.content == ""
        assert turn.message.reasoning == "hm"
        assert turn.finish_reason == "length"
        assert turn.model == MODEL
        assert turn.usage is not None
        assert turn.usage.total_tokens == 7

    async def test_tool_calls_come_back_complete(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "read", "arguments": '{"p":"a"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            )

        turn = await make_provider(handler).complete_non_streaming([Message.user("hi")])
        assert turn.message.tool_calls == (ToolCall("c1", "read", '{"p":"a"}'),)

    async def test_malformed_body_raises_malformed_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>proxy error</html>")

        with pytest.raises(MalformedResponseError):
            await make_provider(handler).complete_non_streaming([Message.user("hi")])

    async def test_complete_uses_the_stream_path(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("accept", ""))
            return sse(
                [
                    content_chunk("x"),
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
            )

        turn = await make_provider(handler).complete([Message.user("hi")])
        assert seen == ["text/event-stream"]
        assert turn.message.content == "x"


# -- token counting --------------------------------------------------------


class TestCountTokens:
    async def test_absent_capability_short_circuits_without_a_request(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json={"count": 1})

        count = await make_provider(handler).count_tokens([Message.user("hi")])
        assert count == TokenCount(None, source="not advertised")
        assert calls == []

    async def test_missing_route_reports_unknown_rather_than_estimating(self) -> None:
        """An estimate would be a made-up number, so we refuse to give one."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "Not Found"})

        provider = make_provider(handler, capabilities=Capabilities(supports_token_api=True))
        count = await provider.count_tokens([Message.user("hi")])
        assert count.tokens is None
        assert count.source == "no /tokenize route"
        assert count.known is False

    async def test_exact_count_from_server(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"count": 1234})

        provider = make_provider(handler, capabilities=Capabilities(supports_token_api=True))
        count = await provider.count_tokens([Message.user("hi")])
        assert count.tokens == 1234
        assert count.source == "server /tokenize"

    async def test_unreachable_tokenizer_reports_unknown(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        provider = make_provider(handler, capabilities=Capabilities(supports_token_api=True))
        count = await provider.count_tokens([Message.user("hi")])
        assert count.tokens is None

    async def test_body_without_a_count_is_malformed_not_zero(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"tokens": [1, 2, 3]})

        provider = make_provider(handler, capabilities=Capabilities(supports_token_api=True))
        with pytest.raises(MalformedResponseError):
            await provider.count_tokens([Message.user("hi")])


# -- health ----------------------------------------------------------------


class TestHealth:
    async def test_models_route_is_used_not_health(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json={"data": [{"id": MODEL}]})

        health = await make_provider(handler).health()
        assert seen == ["/v1/models"]
        assert health.ok
        assert health.models == (MODEL,)
        assert health.latency_ms is not None

    async def test_unreachable_is_reported_not_raised(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        health = await make_provider(handler).health()
        assert health.ok is False
        assert "could not reach" in health.detail

    async def test_empty_model_list_is_not_healthy(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        health = await make_provider(handler).health()
        assert health.ok is False
        assert "no models" in health.detail

    async def test_error_body_is_quoted_back(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503, json={"error": {"message": "engine busy", "code": "overloaded"}}
            )

        health = await make_provider(handler).health()
        assert health.ok is False
        assert "engine busy" in health.detail
        assert "overloaded" in health.detail


# -- construction ----------------------------------------------------------


class TestConstruction:
    def test_trailing_slash_is_normalised(self) -> None:
        provider = OpenAICompatProvider(name="p", base_url=f"{BASE}/", model=MODEL)
        assert provider.base_url == BASE

    def test_capabilities_start_pessimistic(self) -> None:
        provider = OpenAICompatProvider(name="p", base_url=BASE, model=MODEL)
        assert provider.capabilities.supports_tools is False
        assert provider.capabilities.max_model_len is None

    async def test_no_api_key_means_no_authorization_header(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return sse([{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(name="p", base_url=BASE, model=MODEL, client=client)
        await collect(provider, [Message.user("hi")])
        assert "authorization" not in seen[0].headers

    async def test_extra_headers_are_merged(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return sse([{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}])

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = OpenAICompatProvider(
            name="p",
            base_url=BASE,
            model=MODEL,
            extra_headers={"X-Trace": "abc"},
            client=client,
        )
        await collect(provider, [Message.user("hi")])
        assert seen[0].headers["x-trace"] == "abc"

    def test_repr_does_not_leak_the_api_key(self) -> None:
        provider = OpenAICompatProvider(name="p", base_url=BASE, model=MODEL, api_key="hunter2")
        assert "hunter2" not in repr(provider)

"""Wire decoding against payloads captured from the live endpoint.

Every shape asserted here was measured against a real vLLM behind a gateway,
not read off the OpenAI docs. These are the regressions that would otherwise
only show up as a garbled answer at 2am: the usage chunk that carries
``"choices": []``, tool-call arguments split across events, and a reasoning
model that sends ``content: null``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

from bevel.providers._wire import (
    USAGE_GRACE_CHUNKS,
    ChatStreamDecoder,
    iter_sse_data,
    normalise_finish_reason,
    parse_completion_response,
    parse_error_body,
    parse_tool_calls_from_message,
    parse_usage,
)
from bevel.providers.base import (
    APIError,
    ContentDelta,
    MalformedResponseError,
    ReasoningDelta,
    StreamEnd,
)


def decode(payloads: Iterable[str]) -> tuple[list[object], StreamEnd]:
    """Feed payloads through the decoder, returning events and the terminal event."""
    decoder = ChatStreamDecoder()
    events: list[object] = []
    for payload in payloads:
        events.extend(decoder.feed(payload))
    return events, decoder.finalize()


def chunk(**choice: object) -> str:
    return json.dumps({"choices": [choice]})


class TestIterSseData:
    def test_accepts_both_spacing_conventions(self) -> None:
        assert list(iter_sse_data(["data: a", ""])) == ["a"]
        assert list(iter_sse_data(["data:b", ""])) == ["b"]

    def test_ignores_comments_and_non_data_fields(self) -> None:
        lines = [": keepalive", "event: message", "data: x", ""]
        assert list(iter_sse_data(lines)) == ["x"]

    def test_joins_multi_line_events(self) -> None:
        assert list(iter_sse_data(["data: a", "data: b", ""])) == ["a\nb"]

    def test_flushes_trailing_event_without_blank_line(self) -> None:
        assert list(iter_sse_data(["data: last"])) == ["last"]

    def test_blank_line_between_events_does_not_emit_empty_payload(self) -> None:
        assert list(iter_sse_data(["data: a", "", "", "data: b", ""])) == ["a", "b"]


class TestStreamBasics:
    def test_content_deltas_arrive_in_order(self) -> None:
        events, end = decode(
            [
                chunk(delta={"content": "he"}),
                chunk(delta={"content": "llo"}),
                chunk(delta={}, finish_reason="stop"),
                "[DONE]",
            ]
        )
        assert events == [ContentDelta("he"), ContentDelta("llo")]
        assert end.finish_reason == "stop"

    def test_reasoning_arrives_on_its_own_channel(self) -> None:
        """The endpoint streams delta.reasoning before delta.content exists."""
        events, _end = decode(
            [
                chunk(delta={"reasoning": "thinking"}),
                chunk(delta={"content": "answer"}),
            ]
        )
        assert events == [ReasoningDelta("thinking"), ContentDelta("answer")]

    def test_reasoning_content_alias_is_accepted(self) -> None:
        """llama.cpp and some vLLM builds use delta.reasoning_content."""
        events, _end = decode([chunk(delta={"reasoning_content": "hmm"})])
        assert events == [ReasoningDelta("hmm")]

    def test_empty_payload_is_a_keepalive(self) -> None:
        events, end = decode(["", "   "])
        assert events == []
        assert end.finish_reason == "abort"

    def test_done_sentinel_finishes(self) -> None:
        _events, end = decode([chunk(delta={"content": "x"}), "[DONE]"])
        assert end.finish_reason == "abort"  # never given a reason, so not "stop"

    def test_stream_without_finish_reason_is_an_abort(self) -> None:
        """A truncated stream must not be mistaken for a clean stop."""
        _events, end = decode([chunk(delta={"content": "partial"})])
        assert end.finish_reason == "abort"

    def test_unknown_finish_reason_becomes_abort(self) -> None:
        _events, end = decode([chunk(delta={}, finish_reason="eos_token")])
        assert end.finish_reason == "abort"

    def test_malformed_chunk_raises_with_body(self) -> None:
        decoder = ChatStreamDecoder()
        with pytest.raises(MalformedResponseError) as excinfo:
            decoder.feed("{not json")
        assert excinfo.value.body == "{not json"

    def test_choices_must_be_a_list(self) -> None:
        decoder = ChatStreamDecoder()
        with pytest.raises(MalformedResponseError, match="choices"):
            decoder.feed(json.dumps({"choices": {"delta": {}}}))

    def test_error_object_inside_a_200_stream_raises_api_error(self) -> None:
        """Gateways sometimes return errors with a 200 status, inside the stream.

        Same exception type as an out-of-band HTTP error, so the caller has one
        handling path rather than one per transport shape.
        """
        decoder = ChatStreamDecoder()
        with pytest.raises(APIError, match="model not loaded"):
            decoder.feed(json.dumps({"error": {"message": "model not loaded"}}))

    def test_model_is_reported_from_whichever_chunk_carries_it(self) -> None:
        _events, end = decode(
            [
                json.dumps({"model": "qwen", "choices": [{"delta": {"content": "x"}}]}),
                chunk(delta={}, finish_reason="stop"),
            ]
        )
        assert end.model == "qwen"


class TestUsageChunk:
    def test_usage_chunk_has_empty_choices_and_yields_no_events(self) -> None:
        """Measured: the final chunk is {"choices": [], "usage": {...}}.

        Indexing choices[0] unconditionally here is an IndexError on every
        streamed request.
        """
        events, end = decode(
            [
                chunk(delta={"content": "hi"}),
                json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 5,
                            "total_tokens": 16,
                            "completion_tokens_details": {"reasoning_tokens": 3},
                        },
                    }
                ),
                "[DONE]",
            ]
        )
        assert events == [ContentDelta("hi")]
        assert end.usage is not None
        assert end.usage.prompt_tokens == 11
        assert end.usage.completion_tokens == 5
        assert end.usage.reasoning_tokens == 3
        assert end.usage.total_tokens == 16

    def test_missing_choices_key_is_also_tolerated(self) -> None:
        events, _end = decode([json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})])
        assert events == []

    def test_total_is_derived_when_the_server_omits_it(self) -> None:
        usage = parse_usage({"prompt_tokens": 7, "completion_tokens": 4})
        assert usage is not None
        assert usage.total_tokens == 11

    def test_incomplete_usage_is_none_rather_than_zeroes(self) -> None:
        assert parse_usage({"prompt_tokens": 5}) is None
        assert parse_usage(None) is None

    def test_decoder_stays_open_after_the_finish_reason(self) -> None:
        """Measured: vLLM sends the usage chunk *after* the finish_reason chunk.

        Treating a finish reason as "nothing left to read" throws away the token
        count, which is the number the budget guard and the TUI depend on.
        """
        decoder = ChatStreamDecoder()
        decoder.feed(chunk(delta={"content": "hi"}))
        decoder.feed(chunk(delta={}, finish_reason="stop"))
        assert decoder.finished is False

        decoder.feed(
            json.dumps(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
                }
            )
        )
        end = decoder.finalize()
        assert end.usage is not None
        assert end.usage.total_tokens == 16

    def test_decoder_eventually_gives_up_without_done(self) -> None:
        """A gateway that holds the connection open must not hang bevel."""
        decoder = ChatStreamDecoder()
        decoder.feed(chunk(delta={}, finish_reason="stop"))
        for _ in range(USAGE_GRACE_CHUNKS):
            assert decoder.finished is False
            decoder.feed(chunk(delta={"content": "."}))
        assert decoder.finished is True

    def test_done_ends_the_stream_immediately(self) -> None:
        decoder = ChatStreamDecoder()
        decoder.feed(chunk(delta={}, finish_reason="stop"))
        decoder.feed("[DONE]")
        assert decoder.finished is True

    def test_reasoning_tokens_default_to_zero(self) -> None:
        usage = parse_usage({"prompt_tokens": 1, "completion_tokens": 1})
        assert usage is not None
        assert usage.reasoning_tokens == 0


class TestToolCallAccumulation:
    def test_arguments_are_accumulated_per_index(self) -> None:
        """Measured: only the first fragment per index carries id/name."""
        payloads = [
            chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read"},
                        }
                    ]
                }
            ),
            chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"path"'}}]}),
            chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": ': "a.py"}'}}]}),
            chunk(delta={}, finish_reason="tool_calls"),
            "[DONE]",
        ]
        _events, end = decode(payloads)
        assert end.finish_reason == "tool_calls"
        assert len(end.tool_calls) == 1
        call = end.tool_calls[0]
        assert (call.id, call.name, call.arguments) == ("call_a", "read", '{"path": "a.py"}')

    def test_parallel_calls_interleave_without_mixing(self) -> None:
        """Verified working on the endpoint, including interleaved fragments."""
        payloads = [
            chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "id": "a", "function": {"name": "one"}},
                        {"index": 1, "id": "b", "function": {"name": "two"}},
                    ]
                }
            ),
            chunk(delta={"tool_calls": [{"index": 1, "function": {"arguments": '{"y":2}'}}]}),
            chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"x":1}'}}]}),
            chunk(delta={}, finish_reason="tool_calls"),
        ]
        _events, end = decode(payloads)
        assert [(c.id, c.name, c.arguments) for c in end.tool_calls] == [
            ("a", "one", '{"x":1}'),
            ("b", "two", '{"y":2}'),
        ]

    def test_order_follows_first_appearance_not_index_sort(self) -> None:
        payloads = [
            chunk(delta={"tool_calls": [{"index": 3, "id": "late", "function": {"name": "n"}}]}),
            chunk(delta={"tool_calls": [{"index": 0, "id": "early", "function": {"name": "m"}}]}),
        ]
        _events, end = decode(payloads)
        assert [c.id for c in end.tool_calls] == ["late", "early"]

    def test_missing_index_does_not_collide(self) -> None:
        """Some servers omit index; each fragment-less entry still needs its own slot."""
        payloads = [
            chunk(delta={"tool_calls": [{"id": "a", "function": {"name": "one"}}]}),
            chunk(delta={"tool_calls": [{"id": "b", "function": {"name": "two"}}]}),
        ]
        _events, end = decode(payloads)
        assert [c.id for c in end.tool_calls] == ["a", "b"]

    def test_non_string_fragments_are_ignored(self) -> None:
        payloads = [
            chunk(delta={"tool_calls": [{"index": 0, "id": "a", "function": {"name": "n"}}]}),
            chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": None}}]}),
        ]
        _events, end = decode(payloads)
        assert end.tool_calls[0].arguments == ""

    def test_tool_calls_empty_when_none_were_sent(self) -> None:
        _events, end = decode([chunk(delta={"content": "x"}, finish_reason="stop")])
        assert end.tool_calls == ()


class TestNormaliseFinishReason:
    @pytest.mark.parametrize(
        "raw",
        ["stop", "length", "tool_calls", "content_filter"],
    )
    def test_known_reasons_pass_through(self, raw: str) -> None:
        assert normalise_finish_reason(raw) == raw

    @pytest.mark.parametrize("raw", [None, 7, "", "eos_token", "STOP"])
    def test_anything_else_is_abort(self, raw: object) -> None:
        assert normalise_finish_reason(raw) == "abort"


class TestParseErrorBody:
    def test_fastapi_detail_shape(self) -> None:
        """The gateway answers unproxied /v1 routes with exactly this."""
        assert parse_error_body(404, '{"detail":"Not Found"}') == ("Not Found", None)

    def test_openai_error_shape(self) -> None:
        body = '{"error":{"message":"model not found","code":"model_not_found"}}'
        assert parse_error_body(404, body) == ("model not found", "model_not_found")

    def test_fastapi_validation_list_names_the_offending_field(self) -> None:
        """A 422 that only says "field required" is useless without the path."""
        body = json.dumps({"detail": [{"msg": "field required", "loc": ["body", "model"]}]})
        assert parse_error_body(422, body) == ("body.model: field required", None)

    def test_fastapi_validation_list_counts_additional_errors(self) -> None:
        body = json.dumps(
            {
                "detail": [
                    {"msg": "field required", "loc": ["body", "model"]},
                    {"msg": "extra inputs not permitted", "loc": ["body", "top_k"]},
                ]
            }
        )
        assert parse_error_body(422, body) == ("body.model: field required (+1 more)", None)

    def test_fastapi_validation_list_without_a_loc_still_yields_the_msg(self) -> None:
        body = json.dumps({"detail": [{"msg": "value is not a valid integer"}]})
        assert parse_error_body(422, body) == ("value is not a valid integer", None)

    def test_fastapi_validation_list_of_non_dicts_falls_back_to_raw_text(self) -> None:
        body = '{"detail": ["", 5]}'
        assert parse_error_body(422, body) == (body, None)

    def test_plain_text_from_a_proxy_survives(self) -> None:
        assert parse_error_body(502, "Bad Gateway") == ("Bad Gateway", None)

    def test_empty_body_is_described_not_blank(self) -> None:
        message, code = parse_error_body(500, "   ")
        assert message == "empty body with status 500"
        assert code is None

    def test_json_that_is_not_an_object_falls_back_to_text(self) -> None:
        assert parse_error_body(500, "[1,2,3]") == ("[1,2,3]", None)

    def test_oversized_body_is_clipped(self) -> None:
        message, _code = parse_error_body(500, "x" * 5000)
        assert len(message) < 5000
        assert message.endswith("...")

    def test_error_as_a_bare_string(self) -> None:
        assert parse_error_body(400, '{"error":"bad request"}') == ("bad request", None)


class TestParseCompletionResponse:
    def test_reasoning_model_returns_null_content(self) -> None:
        """Measured: content is null, reasoning populated, finish_reason length."""
        body = json.dumps(
            {
                "model": "qwen",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning": "deep thought",
                        },
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 9, "total_tokens": 12},
            }
        )
        content, reasoning, finish, usage, model = parse_completion_response(body)
        assert content == ""
        assert reasoning == "deep thought"
        assert finish == "length"
        assert usage is not None
        assert usage.total_tokens == 12
        assert model == "qwen"

    def test_reasoning_content_alias_works_here_too(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"reasoning_content": "alt"}, "finish_reason": "stop"}]}
        )
        _content, reasoning, _finish, _usage, _model = parse_completion_response(body)
        assert reasoning == "alt"

    def test_missing_finish_reason_is_abort(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "x"}}]})
        assert parse_completion_response(body)[2] == "abort"

    def test_no_choices_is_malformed(self) -> None:
        with pytest.raises(MalformedResponseError, match="no choices"):
            parse_completion_response('{"choices": []}')

    def test_non_json_is_malformed(self) -> None:
        with pytest.raises(MalformedResponseError, match="not JSON"):
            parse_completion_response("<html>502</html>")

    def test_usage_absent_is_none_not_zeroes(self) -> None:
        body = json.dumps({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]})
        assert parse_completion_response(body)[3] is None


class TestParseToolCallsFromMessage:
    def body(self, message: dict[str, object]) -> str:
        return json.dumps({"choices": [{"message": message, "finish_reason": "tool_calls"}]})

    def test_complete_calls_from_a_non_streaming_response(self) -> None:
        """Non-streaming bodies carry each call complete, not in fragments."""
        body = self.body(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"p":"x"}'},
                    },
                    {"id": "b", "function": {"name": "write", "arguments": "{}"}},
                ],
            }
        )
        calls = parse_tool_calls_from_message(body)
        assert [(c.id, c.name, c.arguments) for c in calls] == [
            ("a", "read", '{"p":"x"}'),
            ("b", "write", "{}"),
        ]

    def test_arguments_are_handed_on_unparsed(self) -> None:
        """Malformed JSON in arguments must survive to reach the repair layer."""
        body = self.body(
            {"tool_calls": [{"id": "a", "function": {"name": "n", "arguments": "{oops"}}]}
        )
        assert parse_tool_calls_from_message(body)[0].arguments == "{oops"

    def test_missing_id_gets_a_positional_one(self) -> None:
        body = self.body({"tool_calls": [{"function": {"name": "n", "arguments": "{}"}}]})
        assert parse_tool_calls_from_message(body)[0].id == "call_0"

    def test_absent_or_malformed_tool_calls_yield_nothing(self) -> None:
        assert parse_tool_calls_from_message(self.body({})) == ()
        assert parse_tool_calls_from_message(self.body({"tool_calls": "nope"})) == ()
        assert parse_tool_calls_from_message('{"choices": []}') == ()

    def test_non_json_body_is_malformed(self) -> None:
        with pytest.raises(MalformedResponseError, match="not JSON"):
            parse_tool_calls_from_message("<html>502</html>")

    def test_entries_that_are_not_dicts_are_skipped(self) -> None:
        body = self.body({"tool_calls": ["junk", {"function": {"name": "n", "arguments": "{}"}}]})
        calls = parse_tool_calls_from_message(body)
        assert [c.name for c in calls] == ["n"]

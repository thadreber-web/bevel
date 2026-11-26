"""The JSON-RPC 2.0 core, driven entirely in memory.

m10a exists to answer one question before any transport is written: can
request/response pairing be tested without a pipe? Every test here feeds bytes
to an in-memory :class:`~bevel.mcp.jsonrpc.DuplexStream` and asserts on what
comes back. If these tests needed a subprocess, a socket, or a real pipe, the
abstraction would be wrong and the rest of m10 would be built on it.

The assertions follow the spec's own failure taxonomy rather than our
convenience. A frame that is not JSON, a frame that is JSON but not a legal
request or response, a response whose id matches nothing, and a response
carrying both ``result`` and ``error`` are four different bugs with four
different codes, and each has to be *reported*. A dropped frame is a client
that waits forever for a reply that already arrived broken.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from bevel.mcp.jsonrpc import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    DuplexStream,
    ErrorObject,
    IdAllocator,
    IdExhaustedError,
    InboundNotification,
    InboundRequest,
    JsonRpcChannel,
    MalformedFrame,
    Notification,
    PendingRequest,
    ProtocolViolationError,
    RemoteError,
    Request,
    Response,
    StreamClosedError,
    UnmatchedResponse,
    parse_frame,
)


class FakeStream:
    """An in-memory duplex stream: queued frames in, recorded frames out.

    Deliberately as close to ``{name: b"..."}`` as a Protocol allows. If this
    class ever needs a loop, a thread, or a file descriptor to satisfy
    :class:`DuplexStream`, the Protocol is too wide and m10b is being handed an
    abstraction nobody can test.
    """

    def __init__(self, *incoming: str | bytes) -> None:
        self._incoming: list[bytes] = [
            frame if isinstance(frame, bytes) else frame.encode() for frame in incoming
        ]
        self.written: list[bytes] = []
        self.hang = False

    async def read_frame(self) -> bytes | None:
        if self.hang:
            await asyncio.Event().wait()
        if not self._incoming:
            return None
        return self._incoming.pop(0)

    async def write_frame(self, payload: bytes) -> None:
        self.written.append(payload)

    def push(self, frame: str | bytes) -> None:
        self._incoming.append(frame if isinstance(frame, bytes) else frame.encode())

    def sent(self) -> list[dict[str, Any]]:
        return [json.loads(frame) for frame in self.written]


def response_frame(request_id: Any, **body: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, **body})


def junk_frames() -> list[str]:
    """Frames a hostile or buggy peer could actually put on the wire."""
    return [
        "",
        "   ",
        "not json at all",
        "{",
        "[1, 2, 3]",
        '"a string"',
        "42",
        "true",
        "null",
        "[]",
        "{}",
        '{"jsonrpc": "2.0"}',
        '{"id": 1}',
        '{"jsonrpc": "1.0", "result": 1, "id": 1}',
        '{"jsonrpc": "2.0", "id": null, "result": 1}',
        '{"jsonrpc": "2.0", "id": 1.5, "result": 1}',
        '{"jsonrpc": "2.0", "id": true, "result": 1}',
        '{"jsonrpc": "2.0", "id": 1, "result": 1, "error": {"code": 1, "message": "x"}}',
        '{"jsonrpc": "2.0", "id": 1}',
        '{"jsonrpc": "2.0", "id": 1, "error": null}',
        '{"jsonrpc": "2.0", "id": 1, "error": "nope"}',
        '{"jsonrpc": "2.0", "id": 1, "error": {"code": "one", "message": "x"}}',
        '{"jsonrpc": "2.0", "id": 1, "error": {"code": 1}}',
        '{"jsonrpc": "2.0", "method": 7}',
        '{"jsonrpc": "2.0", "method": ""}',
        '{"jsonrpc": "2.0", "method": "x", "params": 5}',
        '{"jsonrpc": "2.0", "method": "x", "id": null}',
    ]


class TestRequestEnvelope:
    def test_a_request_carries_the_version_id_method_and_params(self) -> None:
        wire = Request(id=7, method="echo", params={"text": "hi"}).to_wire()
        assert wire == {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "echo",
            "params": {"text": "hi"},
        }

    def test_absent_params_are_omitted_rather_than_sent_as_null(self) -> None:
        # The spec permits omitting ``params`` but not sending it as ``null``,
        # and a strict server rejects the latter.
        assert "params" not in Request(id=1, method="ping").to_wire()

    def test_encode_is_utf_8_json(self) -> None:
        encoded = Request(id=1, method="echo", params={"t": "é"}).encode()
        assert json.loads(encoded.decode("utf-8"))["params"] == {"t": "é"}

    def test_positional_params_are_allowed(self) -> None:
        assert Request(id=1, method="m", params=["a", "b"]).to_wire()["params"] == ["a", "b"]

    def test_an_empty_method_is_refused_before_it_reaches_the_wire(self) -> None:
        with pytest.raises(ProtocolViolationError) as excinfo:
            Request(id=1, method="")
        assert excinfo.value.code == INVALID_PARAMS
        assert "method" in str(excinfo.value)

    def test_scalar_params_are_refused(self) -> None:
        # ``params`` must be an object or an array; a bare string is the easy
        # mistake (it *is* a Sequence in Python) and the spec forbids it.
        with pytest.raises(ProtocolViolationError) as excinfo:
            Request(id=1, method="m", params="oops")
        assert excinfo.value.code == INVALID_PARAMS

    def test_unserialisable_params_become_an_error_object_not_a_type_error(self) -> None:
        with pytest.raises(ProtocolViolationError) as excinfo:
            Request(id=1, method="m", params={"x": object()}).encode()
        assert excinfo.value.code == INTERNAL_ERROR
        assert "serialis" in str(excinfo.value)


class TestNotificationEnvelope:
    def test_a_notification_has_no_id(self) -> None:
        wire = Notification(method="progress", params={"n": 1}).to_wire()
        assert wire == {"jsonrpc": "2.0", "method": "progress", "params": {"n": 1}}
        assert "id" not in wire

    def test_encode_is_json(self) -> None:
        assert json.loads(Notification(method="ping").encode()) == {
            "jsonrpc": "2.0",
            "method": "ping",
        }


class TestErrorObject:
    def test_to_wire_omits_absent_data(self) -> None:
        assert ErrorObject(code=-32601, message="Method not found").to_wire() == {
            "code": -32601,
            "message": "Method not found",
        }

    def test_data_is_carried_through(self) -> None:
        wire = ErrorObject(code=-32000, message="boom", data={"retry": True}).to_wire()
        assert wire["data"] == {"retry": True}

    def test_the_named_constructors_carry_their_spec_codes(self) -> None:
        assert ErrorObject.parse_error("bad").code == PARSE_ERROR
        assert ErrorObject.invalid_request("bad").code == INVALID_REQUEST
        assert ErrorObject.method_not_found("nope").code == METHOD_NOT_FOUND
        assert ErrorObject.invalid_params("bad").code == INVALID_PARAMS
        assert ErrorObject.internal_error("bad").code == INTERNAL_ERROR


class TestIdAllocator:
    def test_ids_start_at_one_and_never_repeat(self) -> None:
        allocator = IdAllocator()
        assert [allocator.allocate() for _ in range(3)] == [1, 2, 3]

    def test_two_allocators_do_not_share_a_counter(self) -> None:
        assert IdAllocator().allocate() == IdAllocator().allocate() == 1

    def test_exhaustion_raises_rather_than_reusing_an_id(self) -> None:
        # Wrapping the counter would hand an in-flight id to a second caller,
        # and a response would be delivered to the wrong conversation. Refusing
        # is the only safe answer, so it has to be loud.
        allocator = IdAllocator(limit=2)
        assert allocator.allocate() == 1
        assert allocator.allocate() == 2
        with pytest.raises(IdExhaustedError):
            allocator.allocate()


class TestParseFrame:
    def test_a_well_formed_response_parses(self) -> None:
        frame = parse_frame(response_frame(3, result={"ok": True}))
        assert isinstance(frame, Response)
        assert frame.id == 3
        assert frame.result == {"ok": True}
        assert frame.error is None
        assert frame.is_error is False

    def test_a_result_of_null_is_a_result_and_not_an_absent_one(self) -> None:
        frame = parse_frame(response_frame(3, result=None))
        assert isinstance(frame, Response)
        assert frame.result is None
        assert frame.is_error is False

    def test_malformed_json_is_a_parse_error_naming_the_problem(self) -> None:
        frame = parse_frame('{"jsonrpc": "2.0", "id": 1, ')
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == PARSE_ERROR
        assert "JSON" in frame.error.message
        assert frame.raw.startswith('{"jsonrpc"')

    def test_a_valid_frame_that_is_not_an_object_is_an_invalid_request(self) -> None:
        frame = parse_frame("[1, 2, 3]")
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == INVALID_REQUEST
        assert "array" in frame.error.message

    def test_a_top_level_batch_is_refused_loudly_not_ignored(self) -> None:
        # bevel does not speak batches; saying so beats answering half of them.
        frame = parse_frame('[{"jsonrpc": "2.0", "id": 1, "result": 1}]')
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == INVALID_REQUEST
        assert "batch" in frame.error.message.lower()

    def test_a_response_with_both_result_and_error_is_refused(self) -> None:
        frame = parse_frame(
            '{"jsonrpc": "2.0", "id": 1, "result": 1, "error": {"code": 1, "message": "x"}}'
        )
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == INVALID_REQUEST
        assert "result" in frame.error.message
        assert "error" in frame.error.message

    def test_a_response_with_neither_result_nor_error_is_refused(self) -> None:
        frame = parse_frame('{"jsonrpc": "2.0", "id": 1}')
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == INVALID_REQUEST

    def test_every_problem_in_one_frame_is_reported_at_once(self) -> None:
        # Same house rule as the schema validator: one round trip should name
        # every defect, not the first one.
        frame = parse_frame('{"jsonrpc": "1.0", "id": 1, "result": 1, "error": {}}')
        assert isinstance(frame, MalformedFrame)
        message = frame.error.message
        assert "jsonrpc" in message
        assert "result" in message
        assert "error" in message
        assert "code" in message
        assert "message" in message

    def test_an_error_object_without_a_message_is_refused(self) -> None:
        frame = parse_frame('{"jsonrpc": "2.0", "id": 1, "error": {"code": 1}}')
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == INVALID_REQUEST
        assert "message" in frame.error.message

    def test_a_peer_error_keeps_its_own_code_and_message(self) -> None:
        frame = parse_frame(
            '{"jsonrpc": "2.0", "id": 4, "error": {"code": -32601, "message": "no such method"}}'
        )
        assert isinstance(frame, Response)
        assert frame.is_error is True
        assert frame.error == ErrorObject(code=-32601, message="no such method")

    def test_an_implementation_defined_error_code_survives(self) -> None:
        # -32000..-32099 is reserved for peers; rewriting it would destroy the
        # only information the user can act on.
        frame = parse_frame(
            '{"jsonrpc": "2.0", "id": 4, "error": {"code": -32042, "message": "disk"}}'
        )
        assert isinstance(frame, Response)
        assert frame.error is not None
        assert frame.error.code == -32042

    def test_an_inbound_notification_is_recognised(self) -> None:
        frame = parse_frame('{"jsonrpc": "2.0", "method": "log", "params": {"m": "hi"}}')
        assert isinstance(frame, InboundNotification)
        assert frame.method == "log"
        assert frame.params == {"m": "hi"}

    def test_an_inbound_request_is_recognised_as_distinct_from_a_notification(self) -> None:
        # A peer may send *us* requests. It is legal JSON-RPC, so classifying
        # it as malformed would be a lie; m10c is what answers it.
        frame = parse_frame('{"jsonrpc": "2.0", "id": 9, "method": "ping"}')
        assert isinstance(frame, InboundRequest)
        assert frame.id == 9
        assert frame.method == "ping"

    @pytest.mark.parametrize("frame", junk_frames())
    def test_no_input_ever_raises_out_of_the_parser(self, frame: str) -> None:
        # The parser's contract is total: bytes in, an ``Incoming`` out. A
        # ValueError escaping here is a caller that dies instead of reporting.
        assert parse_frame(frame) is not None

    def test_undecodable_bytes_are_a_parse_error_not_a_unicode_error(self) -> None:
        frame = parse_frame(b"\xff\xfe not utf-8")
        assert isinstance(frame, MalformedFrame)
        assert frame.error.code == PARSE_ERROR


class TestChannelRoundTrip:
    async def test_a_call_writes_one_frame_and_returns_its_result(self) -> None:
        stream = FakeStream(response_frame(1, result="pong"))
        channel = JsonRpcChannel(stream)

        assert await channel.call("ping") == "pong"
        assert stream.sent() == [{"jsonrpc": "2.0", "id": 1, "method": "ping"}]

    async def test_params_reach_the_wire_untouched(self) -> None:
        stream = FakeStream(response_frame(1, result=None))
        channel = JsonRpcChannel(stream)
        await channel.call("echo", {"path": "a.py", "n": 2})
        assert stream.sent()[0]["params"] == {"path": "a.py", "n": 2}

    async def test_ids_increase_across_calls_and_are_never_reused(self) -> None:
        stream = FakeStream(
            response_frame(1, result="a"),
            response_frame(2, result="b"),
            response_frame(3, result="c"),
        )
        channel = JsonRpcChannel(stream)
        assert [await channel.call("m") for _ in range(3)] == ["a", "b", "c"]
        assert [frame["id"] for frame in stream.sent()] == [1, 2, 3]

    async def test_a_notification_writes_a_frame_and_waits_for_nothing(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)

        await channel.notify("cancelled", {"id": 1})

        assert stream.sent() == [{"jsonrpc": "2.0", "method": "cancelled", "params": {"id": 1}}]
        assert channel.pending_ids == ()

    async def test_a_notification_does_not_consume_an_id(self) -> None:
        stream = FakeStream(response_frame(1, result="ok"))
        channel = JsonRpcChannel(stream)
        await channel.notify("hello")
        assert await channel.call("ping") == "ok"
        assert stream.sent()[1]["id"] == 1


class TestPairing:
    async def test_out_of_order_responses_reach_the_caller_that_asked(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)

        slow: PendingRequest = await channel.request("slow")
        fast: PendingRequest = await channel.request("fast")
        assert (slow.id, fast.id) == (1, 2)

        # The peer answers the second request first.
        stream.push(response_frame(2, result="B"))
        stream.push(response_frame(1, result="A"))
        assert isinstance(await channel.receive(), Response)
        assert isinstance(await channel.receive(), Response)

        assert await slow.result() == "A"
        assert await fast.result() == "B"

    async def test_a_response_is_delivered_to_exactly_one_caller(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        pending = await channel.request("slow")
        stream.push(response_frame(1, result="A"))
        await channel.receive()
        assert pending.done is True

        # The peer answers the same id twice. The second answer is not a
        # second delivery, and it is not nothing either.
        stream.push(response_frame(1, result="A again"))
        again = await channel.receive()
        assert isinstance(again, UnmatchedResponse)
        assert again.id == 1

    async def test_an_unknown_id_is_reported_rather_than_dropped(self) -> None:
        stream = FakeStream(response_frame(99, result="ghost"))
        channel = JsonRpcChannel(stream)

        arrived = await channel.receive()

        assert isinstance(arrived, UnmatchedResponse)
        assert arrived.id == 99
        assert "99" in arrived.reason
        assert channel.reports == (arrived,)

    async def test_a_response_to_a_notification_id_is_reported_as_unmatched(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        await channel.notify("hello")
        stream.push(response_frame(1, result="unexpected"))

        arrived = await channel.receive()

        assert isinstance(arrived, UnmatchedResponse)
        assert channel.reports[0] is arrived

    async def test_a_late_duplicate_response_is_reported_not_redelivered(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        pending = await channel.request("slow")
        stream.push(response_frame(1, result="first"))
        await channel.receive()
        stream.push(response_frame(1, result="second"))

        assert isinstance(await channel.receive(), UnmatchedResponse)
        assert await pending.result() == "first"

    async def test_pending_ids_name_the_requests_still_waiting(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        first = await channel.request("a")
        await channel.request("b")
        assert channel.pending_ids == (1, 2)

        stream.push(response_frame(1, result=None))
        await channel.receive()
        assert channel.pending_ids == (2,)
        assert first.done is True


class TestErrorSurfacing:
    async def test_an_error_response_surfaces_its_code_message_and_data(self) -> None:
        stream = FakeStream(
            response_frame(
                1,
                error={"code": -32601, "message": "Method not found", "data": {"hint": "typo"}},
            )
        )
        channel = JsonRpcChannel(stream)

        with pytest.raises(RemoteError) as excinfo:
            await channel.call("nope")

        assert excinfo.value.code == -32601
        assert excinfo.value.message == "Method not found"
        assert excinfo.value.data == {"hint": "typo"}
        assert excinfo.value.request_id == 1
        assert excinfo.value.method == "nope"
        assert "-32601" in str(excinfo.value)
        assert "nope" in str(excinfo.value)

    async def test_an_error_response_resolves_the_pending_request(self) -> None:
        stream = FakeStream(response_frame(1, error={"code": -32000, "message": "boom"}))
        channel = JsonRpcChannel(stream)
        with pytest.raises(RemoteError):
            await channel.call("m")
        assert channel.pending_ids == ()

    async def test_a_malformed_frame_fails_the_waiting_caller_rather_than_hanging_it(self) -> None:
        # The whole point of the slice: a broken reply must not leave a caller
        # blocked on a response that already arrived.
        stream = FakeStream("}{")
        channel = JsonRpcChannel(stream)
        with pytest.raises(ProtocolViolationError) as excinfo:
            await channel.call("m")
        assert excinfo.value.code == PARSE_ERROR
        assert channel.pending_ids == ()

    async def test_a_malformed_frame_is_also_recorded_for_the_operator(self) -> None:
        stream = FakeStream("}{")
        channel = JsonRpcChannel(stream)
        with pytest.raises(ProtocolViolationError):
            await channel.call("m")
        assert isinstance(channel.reports[0], MalformedFrame)

    async def test_an_inbound_request_is_not_mistaken_for_a_reply(self) -> None:
        stream = FakeStream('{"jsonrpc": "2.0", "id": 1, "method": "ping"}')
        channel = JsonRpcChannel(stream)
        arrived = await channel.receive()
        assert isinstance(arrived, InboundRequest)
        assert channel.reports == ()


class TestStreamLifecycle:
    async def test_a_closed_stream_fails_every_pending_request(self) -> None:
        # Failing only the reader would leave the other callers blocked
        # forever, which is the hang this slice exists to prevent.
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        first = await channel.request("a")
        second = await channel.request("b")

        with pytest.raises(StreamClosedError):
            await channel.receive()

        for pending in (first, second):
            with pytest.raises(StreamClosedError):
                await pending.result()
        assert channel.pending_ids == ()

    async def test_a_call_on_a_closed_stream_reports_what_was_waiting(self) -> None:
        stream = FakeStream()
        channel = JsonRpcChannel(stream)
        with pytest.raises(StreamClosedError) as excinfo:
            await channel.call("ping")
        assert "ping" in str(excinfo.value)

    async def test_a_cancelled_call_leaves_nothing_pending(self) -> None:
        stream = FakeStream()
        stream.hang = True
        channel = JsonRpcChannel(stream)
        task = asyncio.create_task(channel.call("slow"))
        await asyncio.sleep(0)
        assert channel.pending_ids == (1,)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert channel.pending_ids == ()


class TestProtocolShape:
    def test_the_fake_satisfies_the_protocol(self) -> None:
        # The Protocol has to be implementable by an object holding queued
        # bytes. If this assertion ever fails, m10b inherited a transport
        # requirement it cannot fake, and the slice is wrong.
        assert isinstance(FakeStream(), DuplexStream)

    def test_a_stream_missing_a_member_is_rejected(self) -> None:
        class ReadOnly:
            async def read_frame(self) -> bytes | None:
                return None

        assert not isinstance(ReadOnly(), DuplexStream)

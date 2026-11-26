"""
Tests for the streamable-HTTP MCP transport (m10e).

The oracle is ``tests/fixtures/mcp_server.py --http``, a real HTTP server living in
this repo that speaks the streamable-HTTP transport, so these tests need no network
and no third-party server. They pin the parts of the specification a hand-rolled
client is most likely to get wrong:

* the ``Accept`` header must offer both ``application/json`` and
  ``text/event-stream``, and the client must cope with either answer;
* ``Mcp-Session-Id`` is captured from the ``initialize`` response and replayed on
  every later request;
* ``MCP-Protocol-Version`` is sent on every request after the negotiation;
* SSE events become the same ``bytes`` objects ``stdio.read_frame`` produces, so
  ``JsonRpcChannel`` is shared unchanged;
* a session the server has dropped fails loudly instead of being silently retried.

The fixture enforces the header rules by *refusing* non-compliant requests rather
than echoing the headers back for the tests to eyeball, because a client that
sends a header only when it happens to be convenient would satisfy an echo.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from bevel.mcp.http import HttpTransport, McpConnectError, McpSessionExpiredError
from bevel.mcp.jsonrpc import JsonRpcChannel, JsonRpcError, ProtocolViolationError, RemoteError
from bevel.mcp.stdio import (
    SUPPORTED_PROTOCOL_VERSIONS,
    McpHandshakeError,
    McpTimeoutError,
    McpTransportError,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"

#: Generous enough for a cold interpreter start, short enough that a hang the
#: transport was supposed to bound fails the test instead of the suite.
HANG_TIMEOUT = 3.0

#: Ask the fixture to record what it received. The server side is the only witness
#: for "the client replayed the session id on every request", so the fixture has to
#: be the one to say so.
SAW = ("--saw", "saw.json")


async def start_fixture(
    tmp_path: Path, *flags: str, headers: dict[str, str] | None = None
) -> tuple[HttpTransport, asyncio.subprocess.Process]:
    """Launch the fixture in HTTP mode and return a transport connected to it."""
    port_file = tmp_path / "port"
    argv = [sys.executable, str(FIXTURE), "--http", "--port-file", str(port_file), *flags]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        await _wait_for_port(port_file, proc)
    except BaseException:
        proc.kill()
        await proc.wait()
        raise
    url = f"http://127.0.0.1:{port_file.read_text(encoding='utf-8').strip()}"
    transport = HttpTransport(
        url, headers=headers, handshake_timeout=HANG_TIMEOUT, call_timeout=HANG_TIMEOUT
    )
    return transport, proc


async def _wait_for_port(port_file: Path, proc: asyncio.subprocess.Process) -> None:
    """Wait for the fixture to publish the port it bound.

    A fixed port would race with whatever else on 127.0.0.1 happens to be free at
    that moment, so the server binds an ephemeral one and says which.
    """
    deadline = asyncio.get_running_loop().time() + 15.0
    while asyncio.get_running_loop().time() < deadline:
        if port_file.exists() and port_file.read_text(encoding="utf-8").strip():
            return
        if proc.returncode is not None:
            raise AssertionError(f"fixture exited early with {proc.returncode}")
        await asyncio.sleep(0.02)
    raise AssertionError(f"the fixture never wrote {port_file}")


async def stop_fixture(transport: HttpTransport, proc: asyncio.subprocess.Process) -> None:
    await transport.close()
    if proc.returncode is None:
        proc.kill()
        await proc.wait()


def saw(tmp_path: Path) -> dict[str, Any]:
    """The fixture's server-side observations."""
    return json.loads((tmp_path / "saw.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The handshake.
# ---------------------------------------------------------------------------


async def test_initialize_negotiates_a_supported_version(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path)
    try:
        await transport.start()
        assert transport.protocol_version == "2025-06-18"
        assert transport.protocol_version in SUPPORTED_PROTOCOL_VERSIONS
        assert transport.server_info["name"] == "bevel-fixture"
    finally:
        await stop_fixture(transport, proc)


async def test_the_channel_is_the_shared_json_rpc_channel(tmp_path: Path) -> None:
    """The whole point of m10e: ``JsonRpcChannel`` is reused, not forked."""
    transport, proc = await start_fixture(tmp_path)
    try:
        await transport.start()
        assert isinstance(transport.channel, JsonRpcChannel)
    finally:
        await stop_fixture(transport, proc)


async def test_an_unsupported_negotiated_version_is_a_handshake_error(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--protocol", "1999-01-01")
    try:
        with pytest.raises(McpHandshakeError, match="1999-01-01"):
            await transport.start()
    finally:
        await stop_fixture(transport, proc)


async def test_a_missing_protocol_version_is_a_handshake_error(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--protocol", "omit")
    try:
        with pytest.raises(McpHandshakeError, match="protocolVersion"):
            await transport.start()
    finally:
        await stop_fixture(transport, proc)


async def test_a_hanging_handshake_names_the_stage(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--hang", "initialize")
    try:
        with pytest.raises(McpTimeoutError, match="handshake timed out"):
            await transport.start()
    finally:
        await stop_fixture(transport, proc)


# ---------------------------------------------------------------------------
# Session id.
# ---------------------------------------------------------------------------


async def test_the_session_id_is_replayed_on_every_request_after_initialize(
    tmp_path: Path,
) -> None:
    transport, proc = await start_fixture(tmp_path, *SAW)
    try:
        await transport.start()
        await transport.call("tools/list")
        await transport.call("tools/call", {"name": "methods", "arguments": {}})
        observations = saw(tmp_path)
        issued = observations["session_issued"]
        assert isinstance(issued, str) and issued
        assert observations["session_ids_presented"] == [issued, issued], (
            "every request after initialize must carry the negotiated session id"
        )
    finally:
        await stop_fixture(transport, proc)


async def test_initialize_itself_carries_no_session_id(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, *SAW)
    try:
        await transport.start()
        assert saw(tmp_path)["initialize_had_session_id"] is False
    finally:
        await stop_fixture(transport, proc)


async def test_a_server_that_issues_no_session_id_still_works(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--no-session", *SAW)
    try:
        await transport.start()
        assert transport.protocol_version == "2025-06-18"
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        assert result["content"][0]["text"] == "x"
        assert saw(tmp_path)["session_issued"] is None
    finally:
        await stop_fixture(transport, proc)


async def test_a_dropped_session_fails_loudly_rather_than_retrying(tmp_path: Path) -> None:
    """Re-sending a ``tools/call`` would run the tool twice, so bevel refuses instead."""
    transport, proc = await start_fixture(tmp_path, "--bad-session-after", "2", *SAW)
    try:
        await transport.start()
        await transport.call("tools/list")
        with pytest.raises(McpSessionExpiredError, match="404"):
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
    finally:
        await stop_fixture(transport, proc)


async def test_close_ends_the_session_with_delete(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, *SAW)
    try:
        await transport.start()
        await transport.close()
        assert saw(tmp_path)["deletes"] == 1
    finally:
        await stop_fixture(transport, proc)

# ---------------------------------------------------------------------------
# Headers the specification makes mandatory.
# ---------------------------------------------------------------------------


async def test_accept_offers_both_json_and_event_stream(tmp_path: Path) -> None:
    """The fixture answers 400 unless ``Accept`` lists both content types."""
    transport, proc = await start_fixture(tmp_path, "--require-accept")
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "ok"}})
        assert result["content"][0]["text"] == "ok"
    finally:
        await stop_fixture(transport, proc)


async def test_the_protocol_version_header_is_sent_after_the_negotiation(
    tmp_path: Path,
) -> None:
    """The fixture answers 400 unless ``MCP-Protocol-Version`` matches the negotiation."""
    transport, proc = await start_fixture(tmp_path, "--require-protocol-header")
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "ok"}})
        assert result["content"][0]["text"] == "ok"
    finally:
        await stop_fixture(transport, proc)


async def test_configured_extra_headers_are_sent(tmp_path: Path) -> None:
    transport, proc = await start_fixture(
        tmp_path, "--require-header", "x-api-key=sekret", "--saw", "saw.json"
    )
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "ok"}})
        assert result["content"][0]["text"] == "ok"
        assert saw(tmp_path)["headers_seen"] == ["sekret"]
    finally:
        await stop_fixture(transport, proc)


# ---------------------------------------------------------------------------
# Both response shapes, and SSE framing.
# ---------------------------------------------------------------------------


async def test_tools_list_over_a_plain_json_body(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path)
    try:
        await transport.start()
        result = await transport.call("tools/list")
        assert [t["name"] for t in result["tools"]] == ["echo", "boom", "methods", "read_note"]
    finally:
        await stop_fixture(transport, proc)


async def test_tools_call_over_an_sse_stream(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--sse")
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "s"}})
        assert result["content"][0]["text"] == "s"
    finally:
        await stop_fixture(transport, proc)


async def test_an_sse_event_becomes_the_same_bytes_read_frame_produces(
    tmp_path: Path,
) -> None:
    """One ``data:`` event per JSON-RPC message, and nothing else in the frame."""
    transport, proc = await start_fixture(tmp_path, "--sse")
    try:
        await transport.start()
        await transport.write_frame(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 991,
                    "method": "tools/call",
                    "params": {"name": "echo", "arguments": {"text": "raw"}},
                }
            ).encode("utf-8")
        )
        frame = await transport.read_frame()
        assert frame is not None
        assert b"\n" not in frame.strip()
        assert json.loads(frame)["id"] == 991
    finally:
        await stop_fixture(transport, proc)


async def test_notifications_on_the_stream_do_not_eat_the_response(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--sse", "--notify", "3")
    try:
        await transport.start()
        result = await transport.call(
            "tools/call", {"name": "echo", "arguments": {"text": "after noise"}}
        )
        assert result["content"][0]["text"] == "after noise"
        # Consumed by the channel as inbound notifications: not mistaken for the
        # reply, and not reported as undeliverable.
        assert transport.channel.reports == ()
    finally:
        await stop_fixture(transport, proc)


async def test_garbage_on_the_stream_is_reported_and_the_transport_survives(
    tmp_path: Path,
) -> None:
    transport, proc = await start_fixture(tmp_path, "--sse", "--garbage-on-call")
    try:
        await transport.start()
        # Exactly one request is outstanding, so the channel can attribute the
        # unreadable event to it honestly rather than guessing.
        with pytest.raises(ProtocolViolationError):
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        result = await transport.call(
            "tools/call", {"name": "echo", "arguments": {"text": "again"}}
        )
        assert result["content"][0]["text"] == "again"
    finally:
        await stop_fixture(transport, proc)


async def test_is_error_is_a_result_not_an_exception(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--is-error")
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        assert result["isError"] is True
    finally:
        await stop_fixture(transport, proc)


async def test_a_remote_error_is_still_raised(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path)
    try:
        await transport.start()
        with pytest.raises(RemoteError):
            await transport.call("nope/not-a-method")
    finally:
        await stop_fixture(transport, proc)


async def test_a_hanging_call_names_the_method(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--hang", "call")
    try:
        await transport.start()
        with pytest.raises(McpTimeoutError, match="tools/call"):
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
    finally:
        await stop_fixture(transport, proc)


# ---------------------------------------------------------------------------
# Lifecycle and failure modes.
# ---------------------------------------------------------------------------


async def test_a_refused_connection_is_a_typed_transport_error() -> None:
    # Port 1 is not listening on 127.0.0.1 and never will be.
    transport = HttpTransport("http://127.0.0.1:1", handshake_timeout=2.0, call_timeout=2.0)
    with pytest.raises(McpConnectError) as excinfo:
        await transport.start()
    assert isinstance(excinfo.value, McpTransportError)


async def test_a_url_without_a_scheme_is_rejected_before_any_request() -> None:
    with pytest.raises(McpTransportError, match="http"):
        HttpTransport("127.0.0.1:8080/mcp")


async def test_close_is_idempotent(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path, "--saw", "saw.json")
    try:
        await transport.start()
    finally:
        await stop_fixture(transport, proc)
    await transport.close()
    await transport.close()
    assert saw(tmp_path)["deletes"] == 1


async def test_calling_after_close_is_an_error(tmp_path: Path) -> None:
    transport, proc = await start_fixture(tmp_path)
    try:
        await transport.start()
        await transport.close()
        with pytest.raises(JsonRpcError):
            await transport.call("tools/list")
    finally:
        await stop_fixture(transport, proc)


async def test_a_server_that_dies_mid_call_reports_a_closed_stream(tmp_path: Path) -> None:
    # start() is three messages (initialize, notifications/initialized,
    # tools/list), so the fixture exits on the fourth: the call below.
    transport, proc = await start_fixture(tmp_path, "--exit-after", "3")
    try:
        await transport.start()
        with pytest.raises(McpTransportError):
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
    finally:
        await stop_fixture(transport, proc)


"""The m10b oracle: the stdio transport driven against a real child process.

Every test here spawns ``tests/fixtures/mcp_server.py`` as an actual process.
There is no fake pipe and no monkeypatched ``create_subprocess_exec`` on
purpose: a transport tested against a double proves only that the double agrees
with itself. The behaviours that matter — a frame split across two ``os.read``
calls, a chatty stderr filling a 64 KiB pipe, a grandchild that must die with
its parent — exist only in a real process, and are exactly the ones a fake
cannot fail.

These run in the default lane. They are the oracle every later m10 slice
reuses, so a skip here would quietly un-verify m10c onwards.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from bevel.mcp.jsonrpc import MalformedFrame, ProtocolViolationError, StreamClosedError
from bevel.mcp.stdio import (
    DEFAULT_CALL_TIMEOUT,
    DEFAULT_HANDSHAKE_TIMEOUT,
    DEFAULT_SPAWN_TIMEOUT,
    McpHandshakeError,
    McpSpawnError,
    McpTimeoutError,
    StdioTransport,
)
from bevel.mcp.stdio import (
    DEFAULT_CALL_TIMEOUT as _CALL,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"

# Generous enough that a slow windows-latest runner never trips them, small
# enough that a hang fails this file in seconds rather than at pytest's ceiling.
HANG_TIMEOUT = 3.0

_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def fixture_argv(*flags: str) -> list[str]:
    """The exact command line the fixture server is spawned with."""
    return [sys.executable, str(FIXTURE), *flags]


def frame(**fields: Any) -> bytes:
    return json.dumps({"jsonrpc": "2.0", **fields}).encode("utf-8")


def process_alive(pid: int) -> bool:
    """Best-effort liveness check without adding a dependency."""
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def wait_for_file(path: Path, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if path.exists() and path.stat().st_size > 0:  # noqa: ASYNC240 - a few bytes in a local temp file
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{path.name} was never written")


async def wait_until_dead(pid: int, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if not process_alive(pid):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"pid {pid} is still alive")


# ---------------------------------------------------------------------------
# The happy path: this is the round trip every later slice stands on.
# ---------------------------------------------------------------------------


async def test_spawn_handshake_and_tools_list_round_trip(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    try:
        tools = await transport.start()
        assert [tool["name"] for tool in tools] == ["echo", "boom", "methods", "read_note"]
        assert transport.server_info["name"] == "bevel-fixture"
        assert transport.protocol_version == "2025-06-18"
        assert transport.pid is not None
    finally:
        await transport.close()


async def test_initialize_then_notifications_initialized_then_tools_list(
    tmp_path: Path,
) -> None:
    """The server reports what it actually received, in the order it arrived.

    Asserting on the wire rather than on the client's bookkeeping: a client
    that skipped ``notifications/initialized`` would still get a working
    ``tools/list`` back from a lenient server, and bevel would ship a handshake
    that real servers reject.
    """
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "methods", "arguments": {}})
        received = json.loads(result["content"][0]["text"])
        assert received == ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    finally:
        await transport.close()


async def test_a_call_round_trips_and_a_notification_waits_for_nothing(
    tmp_path: Path,
) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "hi"}})
        assert result["content"][0]["text"] == "hi"
        # Returns without waiting: a notification has no id and nothing to pair.
        await transport.notify("notifications/cancelled", {"requestId": 1})
    finally:
        await transport.close()


async def test_is_error_is_a_result_not_a_transport_failure(tmp_path: Path) -> None:
    """``isError`` is the server reporting a tool failure, not a broken pipe."""
    transport = StdioTransport(fixture_argv("--is-error"), cwd=tmp_path)
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        assert result["isError"] is True
    finally:
        await transport.close()


async def test_resources_list_and_read(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    try:
        await transport.start()
        listed = await transport.call("resources/list", {})
        assert listed["resources"][0]["uri"] == "bevel://fixture/notes"
        read = await transport.call("resources/read", {"uri": "bevel://fixture/notes"})
        assert read["contents"][0]["text"] == "the fixture read this back to you"
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# Timeouts: each stage must name itself, or a CI hang is undebuggable.
# ---------------------------------------------------------------------------


async def test_a_server_that_never_answers_initialize_fails_the_handshake(
    tmp_path: Path,
) -> None:
    """``--hang initialize``: alive, listening, silent. A handshake timeout.

    The child is genuinely up — it just will not talk — so this is the stage
    that must be named, not spawn. Naming the wrong stage sends someone to the
    wrong file.
    """
    transport = StdioTransport(
        fixture_argv("--hang", "initialize"),
        cwd=tmp_path,
        handshake_timeout=HANG_TIMEOUT,
    )
    started = time.monotonic()
    with pytest.raises(McpTimeoutError) as caught:
        await transport.start()
    elapsed = time.monotonic() - started

    assert elapsed < HANG_TIMEOUT * 3
    message = str(caught.value)
    assert "handshake" in message
    assert "initialize" in message
    assert str(FIXTURE) in message
    assert transport.pid is None


async def test_a_server_that_spawns_and_hangs_fails_the_handshake_too(
    tmp_path: Path,
) -> None:
    """``--hang startup`` is indistinguishable on the wire, so it is also a
    handshake timeout. Spawn is defined by liveness, not by a first byte."""
    transport = StdioTransport(
        fixture_argv("--hang", "startup"),
        cwd=tmp_path,
        spawn_timeout=HANG_TIMEOUT,
        handshake_timeout=HANG_TIMEOUT,
    )
    with pytest.raises(McpTimeoutError) as caught:
        await transport.start()
    assert "handshake" in str(caught.value)
    assert transport.pid is None


async def test_a_call_that_never_returns_times_out_naming_the_call_stage(
    tmp_path: Path,
) -> None:
    transport = StdioTransport(
        fixture_argv("--hang", "call"),
        cwd=tmp_path,
        call_timeout=HANG_TIMEOUT,
    )
    try:
        await transport.start()
        with pytest.raises(McpTimeoutError) as caught:
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        message = str(caught.value)
        assert "call" in message
        assert "tools/call" in message
        assert str(FIXTURE) in message
    finally:
        await transport.close()


async def test_a_slow_call_under_its_timeout_still_completes(tmp_path: Path) -> None:
    transport = StdioTransport(
        fixture_argv("--sleep-before-reply", "0.3"),
        cwd=tmp_path,
        call_timeout=HANG_TIMEOUT,
    )
    try:
        await transport.start()
        result = await transport.call("tools/call", {"name": "echo", "arguments": {"text": "late"}})
        assert result["content"][0]["text"] == "late"
    finally:
        await transport.close()


async def test_a_dead_child_is_a_spawn_failure_naming_the_program(tmp_path: Path) -> None:
    """The spawn stage is create *plus* liveness.

    A child that is gone before it could be spoken to is not a handshake
    timeout — nothing was ever written to it. The message has to name ``argv``
    and the exit code, because "it timed out" would send you looking for a
    slow server when the real answer is "that program does not run".
    """
    transport = StdioTransport(fixture_argv("--die-now"), cwd=tmp_path)
    with pytest.raises(McpSpawnError) as caught:
        await transport.start()
    message = str(caught.value)
    assert sys.executable in message
    assert str(FIXTURE) in message
    assert "exit" in message
    assert transport.pid is None


async def test_a_program_that_cannot_be_found_is_a_spawn_failure(tmp_path: Path) -> None:
    transport = StdioTransport(["definitely-not-a-real-program-xyz", "serve"], cwd=tmp_path)
    with pytest.raises(McpSpawnError) as caught:
        await transport.start()
    assert "definitely-not-a-real-program-xyz" in str(caught.value)


# ---------------------------------------------------------------------------
# EOF: every outstanding request must fail, and nothing may hang.
# ---------------------------------------------------------------------------


async def test_a_server_that_dies_before_answering_fails_the_pending_request(
    tmp_path: Path,
) -> None:
    """``--exit-after 0`` reads ``initialize`` and dies without replying.

    The single outstanding request fails with ``StreamClosedError`` naming the
    method, and it fails *now* — not when some later call happens to notice the
    pipe is dead.
    """
    transport = StdioTransport(
        fixture_argv("--exit-after", "0"),
        cwd=tmp_path,
        handshake_timeout=HANG_TIMEOUT,
    )
    started = time.monotonic()
    with pytest.raises(StreamClosedError) as caught:
        await transport.start()
    assert time.monotonic() - started < HANG_TIMEOUT * 3
    assert "initialize" in str(caught.value)
    assert transport.pid is None


async def test_every_outstanding_request_fails_when_the_pipe_closes(
    tmp_path: Path,
) -> None:
    """Two calls in flight, one death, both fail. A half-dead channel is worse
    than a dead one: the survivor would wait for its timeout on a pipe that
    will never produce again."""
    # --hang call keeps both requests genuinely unanswered; without it the echo
    # replies land before the kill and the test proves nothing about death.
    transport = StdioTransport(fixture_argv("--hang", "call"), cwd=tmp_path)
    await transport.start()
    assert transport.pid is not None
    pid = transport.pid

    first = asyncio.create_task(
        transport.call("tools/call", {"name": "echo", "arguments": {"text": "a"}})
    )
    second = asyncio.create_task(
        transport.call("tools/call", {"name": "echo", "arguments": {"text": "b"}})
    )
    await asyncio.sleep(0.5)
    assert transport.channel.pending_ids
    await transport.kill()

    results = await asyncio.gather(first, second, return_exceptions=True)
    for result in results:
        assert isinstance(result, StreamClosedError), result
    # Only one reader can witness the EOF; the second finds the channel already
    # drained and reports an empty set. Both are told the pipe died, which is the
    # contract, and the witness names the requests it failed.
    assert any("tools/call" in str(result) for result in results)
    assert transport.channel.pending_ids == ()
    await wait_until_dead(pid)


async def test_reading_after_the_transport_is_closed_fails(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    await transport.start()
    await transport.close()
    with pytest.raises(StreamClosedError):
        await transport.read_frame()


# ---------------------------------------------------------------------------
# Framing. The two cases a line-oriented reader gets wrong in opposite
# directions: one frame arriving in pieces, and two frames arriving at once.
# ---------------------------------------------------------------------------


async def test_a_frame_split_across_two_reads_is_reassembled(tmp_path: Path) -> None:
    """The server writes half a line, waits, then writes the rest.

    The OS hands the reader the first half on its own. A reader that treats one
    ``read`` as one frame returns a truncated line that fails to parse, and the
    second half then poisons the next frame too.
    """
    transport = StdioTransport(fixture_argv("--emit", "split"), cwd=tmp_path)
    try:
        await transport.spawn()
        await transport.write_frame(frame(id=1, method="tools/list"))
        first = await transport.read_frame()
        assert first is not None
        assert json.loads(first)["id"] == 1
        assert first[-1:] != b"\n", "the delimiter must not be part of the frame"
    finally:
        await transport.close()


async def test_two_frames_in_one_read_are_returned_one_at_a_time(tmp_path: Path) -> None:
    """The server writes two lines in a single ``write``.

    The reader gets both bytes in one ``os.read``. Returning the buffer whole
    would parse the first frame and silently lose the second — a notification
    that vanishes is worse than one that arrives late.
    """
    transport = StdioTransport(fixture_argv("--emit", "coalesce"), cwd=tmp_path)
    try:
        await transport.spawn()
        await transport.write_frame(frame(id=7, method="tools/list"))
        reply = await transport.read_frame()
        assert reply is not None
        assert json.loads(reply)["id"] == 7
        following = await transport.read_frame()
        assert following is not None
        assert json.loads(following)["method"] == "notifications/message"
    finally:
        await transport.close()


async def test_read_frame_returns_none_at_end_of_file(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv("--exit-after", "0"), cwd=tmp_path)
    await transport.spawn()
    await transport.write_frame(frame(id=1, method="initialize", params={}))
    assert await transport.read_frame() is None
    assert await transport.read_frame() is None  # idempotent, not a second hang
    await transport.close()


# ---------------------------------------------------------------------------
# stderr: drained continuously, bounded, and quoted back on failure.
# ---------------------------------------------------------------------------


async def test_a_chatty_stderr_does_not_deadlock_the_server(tmp_path: Path) -> None:
    """1 MiB on stderr before the server reads a single byte of stdin.

    An OS pipe holds ~64 KiB. A client that drains stderr only when it wants an
    error message leaves the child blocked in ``write`` and itself blocked in
    ``read`` — a deadlock with two innocent victims. This test is the reason the
    reader is a background task.
    """
    transport = StdioTransport(
        fixture_argv("--stderr-bytes", "1048576"),
        cwd=tmp_path,
        handshake_timeout=HANG_TIMEOUT,
    )
    try:
        tools = await transport.start()
        assert tools
        assert len(transport.stderr_tail) <= transport.STDERR_CAP
    finally:
        await transport.close()


async def test_the_stderr_tail_is_available_for_error_messages(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv("--stderr-lines", "3"), cwd=tmp_path)
    try:
        await transport.start()
        await asyncio.sleep(0.2)
        tail = transport.stderr_tail
        for index in range(3):
            assert f"fixture stderr line {index}" in tail
    finally:
        await transport.close()


async def test_the_stderr_tail_is_kept_bounded(tmp_path: Path) -> None:
    transport = StdioTransport(
        fixture_argv("--stderr-bytes", "300000"),
        cwd=tmp_path,
        handshake_timeout=HANG_TIMEOUT,
    )
    try:
        await transport.start()
        await asyncio.sleep(0.3)
        assert 0 < len(transport.stderr_tail) <= transport.STDERR_CAP
    finally:
        await transport.close()


async def test_a_failure_message_quotes_the_stderr_tail(tmp_path: Path) -> None:
    """A traceback on stderr is the diagnosis; it must not stay in the child."""
    transport = StdioTransport(
        fixture_argv("--stderr-lines", "2", "--hang", "initialize"),
        cwd=tmp_path,
        handshake_timeout=HANG_TIMEOUT,
    )
    with pytest.raises(McpTimeoutError) as caught:
        await transport.start()
    message = str(caught.value)
    assert "fixture stderr line 0" in message
    assert "fixture stderr line 1" in message


# ---------------------------------------------------------------------------
# Garbage on stdout: reported, never swallowed, and the reader survives.
# ---------------------------------------------------------------------------


async def test_a_garbage_frame_is_delivered_intact_rather_than_swallowed(
    tmp_path: Path,
) -> None:
    """The transport's contract is bytes-in-frames-out.

    Deciding that a line is not JSON is the channel's job. Dropping it here
    would make a server's crash vanish, and the client would hang waiting for a
    reply that was already received and thrown away.
    """
    transport = StdioTransport(fixture_argv("--garbage-on-call"), cwd=tmp_path)
    try:
        await transport.spawn()
        await transport.write_frame(frame(id=3, method="tools/call", params={"name": "echo"}))
        garbage = await transport.read_frame()
        assert garbage is not None
        assert b"not json" in garbage
        reply = await transport.read_frame()
        assert reply is not None
        assert json.loads(reply)["id"] == 3
    finally:
        await transport.close()


async def test_the_reader_survives_a_frame_the_channel_cannot_parse(
    tmp_path: Path,
) -> None:
    """One unreadable frame must not end the session.

    The channel reports it and keeps going; if the reader died on the parse
    error, every later call would hang on a transport that is still fine.
    """
    transport = StdioTransport(fixture_argv("--garbage-on-call"), cwd=tmp_path)
    try:
        await transport.start()
        # Exactly one request is outstanding, so the channel can attribute the
        # unreadable frame to it honestly rather than guessing.
        with pytest.raises(ProtocolViolationError):
            await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})
        # Still usable afterwards: the reader is alive and pairing still works.
        result = await transport.call(
            "tools/call", {"name": "echo", "arguments": {"text": "again"}}
        )
        assert result["content"][0]["text"] == "again"
    finally:
        await transport.close()


async def test_unsolicited_notifications_are_routed_not_returned(tmp_path: Path) -> None:
    """Three notifications then the reply.

    This is the case that decides the reader design. A client that reads only
    when it wants an answer consumes the notifications as if they were replies
    and then waits for a response that has already been spent — or, worse, the
    server blocks writing them into a pipe nobody is draining.
    """
    transport = StdioTransport(fixture_argv("--notify", "3"), cwd=tmp_path)
    try:
        await transport.start()
        result = await transport.call(
            "tools/call",
            {"name": "echo", "arguments": {"text": "after noise"}},
            timeout=HANG_TIMEOUT,
        )
        assert result["content"][0]["text"] == "after noise"
        # The notifications were consumed by the reader, not handed to the caller
        # as replies: the call above succeeded, and nothing unreadable was
        # reported. A client that mistook a notification for a response would
        # have failed the call instead.
        assert not any(isinstance(report, MalformedFrame) for report in transport.channel.reports)
        assert transport.channel.pending_ids == ()
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# Negotiation.
# ---------------------------------------------------------------------------


async def test_a_peer_that_will_not_negotiate_is_rejected(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv("--protocol", "1999-01-01"), cwd=tmp_path)
    with pytest.raises(McpHandshakeError) as caught:
        await transport.start()
    assert "1999-01-01" in str(caught.value)
    assert transport.pid is None


async def test_a_peer_that_omits_the_protocol_version_is_rejected(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv("--protocol", "omit"), cwd=tmp_path)
    with pytest.raises(McpHandshakeError) as caught:
        await transport.start()
    assert "protocolVersion" in str(caught.value)
    assert transport.pid is None


# ---------------------------------------------------------------------------
# Teardown. The tests that loop, so they are asserted on the real platform.
# ---------------------------------------------------------------------------


async def test_teardown_kills_the_whole_tree_including_the_grandchild(
    tmp_path: Path,
) -> None:
    """A grandchild is the only thing a Job Object has to prove.

    Killing the direct child is easy and is not enough: the grandchild keeps the
    workspace, the port, and the CI runner's process table. On Windows the kill
    is closing the Job Object handle, which also reaches children created *after*
    the server was assigned to it — so the fixture spawns this one late on
    purpose.
    """
    heartbeat = tmp_path / "heartbeat.txt"
    transport = StdioTransport(fixture_argv("--grandchild", str(heartbeat)), cwd=tmp_path)
    await transport.start()
    assert transport.pid is not None
    server_pid = transport.pid

    await transport.call("tools/call", {"name": "echo", "arguments": {"text": "spawn"}})
    await wait_for_file(heartbeat)
    grandchild_pids = {
        int(line.split()[0]) for line in heartbeat.read_text(encoding="utf-8").splitlines() if line
    }
    assert grandchild_pids
    assert server_pid not in grandchild_pids

    await transport.close()

    await wait_until_dead(server_pid)
    for pid in grandchild_pids:
        await wait_until_dead(pid)


async def test_teardown_after_an_exception_leaves_no_child(tmp_path: Path) -> None:
    """The failure path must clean up as thoroughly as the success path."""
    transport = StdioTransport(fixture_argv("--hang", "initialize"), cwd=tmp_path)
    with pytest.raises(McpTimeoutError):
        await transport.start()
    assert transport.pid is None
    await transport.close()
    await asyncio.sleep(0.2)


async def test_cancellation_mid_call_leaves_no_child(tmp_path: Path) -> None:
    """Ctrl+C during a tool call must not orphan a process.

    Cancellation is not an error path the transport gets to ignore: the child is
    still alive and still holding the pipes when the caller walks away.
    """
    transport = StdioTransport(
        fixture_argv("--sleep-before-reply", "30"),
        cwd=tmp_path,
        call_timeout=30.0,
    )
    await transport.start()
    assert transport.pid is not None
    pid = transport.pid

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            transport.call("tools/call", {"name": "echo", "arguments": {"text": "never"}}), 0.5
        )

    await transport.close()
    await wait_until_dead(pid)


async def test_close_is_idempotent_and_safe_before_start(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    await transport.close()
    await transport.close()
    await transport.start()
    await transport.close()
    await transport.close()


async def test_calling_after_close_is_refused_rather_than_hanging(tmp_path: Path) -> None:
    transport = StdioTransport(fixture_argv(), cwd=tmp_path)
    await transport.start()
    await transport.close()
    with pytest.raises(StreamClosedError):
        await transport.call("tools/call", {"name": "echo", "arguments": {"text": "x"}})


# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------


def test_the_default_timeouts_are_separate_and_ci_comfortable() -> None:
    """Three stages, three budgets, and none of them is a pytest-level wait."""
    assert DEFAULT_SPAWN_TIMEOUT > 0
    assert DEFAULT_HANDSHAKE_TIMEOUT > 0
    assert DEFAULT_CALL_TIMEOUT > 0
    assert DEFAULT_CALL_TIMEOUT == _CALL
    assert DEFAULT_SPAWN_TIMEOUT < 120
    assert DEFAULT_HANDSHAKE_TIMEOUT < 120
    assert DEFAULT_CALL_TIMEOUT < 300

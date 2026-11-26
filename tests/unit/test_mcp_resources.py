"""The m10f oracle: ``resources/list`` / ``resources/read``, and what consumes them.

Every test here drives ``tests/fixtures/mcp_server.py`` as a real child process
(stdio) or a real socket (HTTP, ``--http``). There is no fake transport, in
keeping with the rest of the package: the behaviours that matter -- a server that
goes silent mid-read, a resource too big to hand to a model, a base64 blob that
must not be smuggled into the transcript -- only exist on the far side of a real
pipe, and a double agrees with whatever it was told.

The failure modes are fixture flags, not monkeypatches of the client:
``--hang resources-list``, ``--hang resources-read``, ``--resources-list
{empty,malformed,junk}``, ``--resources-read {large,blob,error,gone,empty}``.

What this file holds to account:

* **Nothing a model can call is ever an empty string.** A server that is down,
  disabled, never handshook, closed, or genuinely empty each get their own
  sentence. "No resources" and "I could not ask" must not look the same, because
  the model's next action differs completely between the two.
* **The two tools are global.** N configured servers cost two offered tools, not
  2N, and neither ever enters ``core/tools/registry.py`` --
  ``test_tools_registry.py`` pins the built-in list, and a remote server must not
  be able to change what that static list says.
* **Both are ``READ`` and parallel-safe**, per PLAN: the server list is config the
  user wrote, not a target the model picked.
* **Content is capped and the cap announces itself**, in the style of
  ``core/tools/shell.py``'s ``DEFAULT_MAX_OUTPUT_BYTES``.
* **A closed transport is not resurrectable.** After ``aclose`` no name resolves.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import McpServerConfig
from bevel.core.agent import Agent
from bevel.core.policy import allow_all_policy
from bevel.core.tools import registry
from bevel.core.tools.base import RiskTier, Tool, ToolContext, ToolResult, call_tool
from bevel.core.tools.schema import unsupported_keywords
from bevel.mcp.http import HttpTransport
from bevel.mcp.jsonrpc import RemoteError
from bevel.mcp.resources import (
    MAX_RESOURCE_CHARS,
    TRUNCATION_MARKER,
    ResourceContent,
    ResourceRef,
    list_resources,
    read_resource,
    resource_tools,
)
from bevel.mcp.servers import McpServers, start_servers
from bevel.mcp.stdio import McpHandshakeError, McpTimeoutError, StdioTransport
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"

#: Generous enough for a cold interpreter on windows-latest, short enough that a
#: hang the slice was supposed to bound fails this file rather than the suite.
HANG_TIMEOUT = 3.0

#: Mirrors in the fixture, restated rather than imported: there is no ``tests``
#: package to import from, so a drift here fails loudly instead of silently.
RESOURCE_URI = "bevel://fixture/notes"
RESOURCE_TEXT = "the fixture read this back to you"
GONE_URI = "bevel://fixture/gone"
BLOB_URI = "bevel://fixture/blob"
#: What the fixture's blob decodes to. If it ever appears in a result, bevel
#: rendered binary into the transcript.
BLOB_SECRET = "SECRET-BINARY-PAYLOAD"

BUILTIN_TOOLS = ["read", "grep", "glob", "recall", "todo", "write", "edit", "remember", "shell"]


def fixture_argv(*flags: str) -> list[str]:
    return [sys.executable, str(FIXTURE), *flags]


def fixture_config(*flags: str, **overrides: Any) -> McpServerConfig:
    fields: dict[str, Any] = {"command": sys.executable, "args": (str(FIXTURE), *flags)}
    fields.update(overrides)
    return McpServerConfig(**fields)


async def _transport(tmp_path: Path, *flags: str) -> StdioTransport:
    transport = StdioTransport(fixture_argv(*flags), cwd=tmp_path, call_timeout=HANG_TIMEOUT)
    await transport.start()
    return transport


async def _started(*flags: str, name: str = "fixture", **overrides: Any) -> McpServers:
    return await start_servers({name: fixture_config(*flags, **overrides)})


def _tool(servers: McpServers, name: str) -> Tool:
    offered = {tool.spec.name: tool for tool in resource_tools(servers)}
    assert name in offered, f"{name} was not offered to the model"
    return offered[name]


async def _call(servers: McpServers, name: str, **arguments: Any) -> ToolResult:
    return await call_tool(_tool(servers, name), arguments, ToolContext(workspace=Path.cwd()))


# ---------------------------------------------------------------------------
# The wire side, over stdio.
# ---------------------------------------------------------------------------


async def test_list_resources_returns_typed_refs(tmp_path: Path) -> None:
    transport = await _transport(tmp_path)
    try:
        refs = await list_resources(transport)
    finally:
        await transport.close()
    assert refs == (
        ResourceRef(
            uri=RESOURCE_URI,
            name="notes",
            description="The fixture's one resource.",
            mime_type="text/plain",
        ),
    )


async def test_read_resource_returns_the_text(tmp_path: Path) -> None:
    transport = await _transport(tmp_path)
    try:
        content = await read_resource(transport, RESOURCE_URI)
    finally:
        await transport.close()
    assert content == ResourceContent(uri=RESOURCE_URI, mime_type="text/plain", text=RESOURCE_TEXT)
    assert content.truncated is False
    assert content.is_error is False


async def test_a_hanging_resources_list_times_out_naming_the_method(tmp_path: Path) -> None:
    """A server that goes silent mid-list must cost a timeout, not the run.

    Reached the same way HTTP reaches a hang -- a handler that emits nothing --
    so there is exactly one hang implementation to keep in sync.
    """
    transport = await _transport(tmp_path, "--hang", "resources-list")
    try:
        with pytest.raises(McpTimeoutError) as caught:
            await list_resources(transport)
        assert "resources/list" in str(caught.value)
    finally:
        await transport.close()


async def test_a_hanging_resources_read_times_out_naming_the_method(tmp_path: Path) -> None:
    transport = await _transport(tmp_path, "--hang", "resources-read")
    try:
        with pytest.raises(McpTimeoutError) as caught:
            await read_resource(transport, RESOURCE_URI)
        assert "resources/read" in str(caught.value)
    finally:
        await transport.close()


async def test_an_empty_list_is_an_empty_tuple_not_an_error(tmp_path: Path) -> None:
    """A server that honestly has nothing is a success, and says so upstream."""
    transport = await _transport(tmp_path, "--resources-list", "empty")
    try:
        assert await list_resources(transport) == ()
    finally:
        await transport.close()


async def test_a_list_with_no_resources_array_is_a_typed_error(tmp_path: Path) -> None:
    transport = await _transport(tmp_path, "--resources-list", "malformed")
    try:
        with pytest.raises(McpHandshakeError) as caught:
            await list_resources(transport)
        assert "resources/list" in str(caught.value)
    finally:
        await transport.close()


async def test_one_broken_entry_costs_itself_and_not_its_siblings(tmp_path: Path) -> None:
    """A server with one malformed descriptor still gives you the good ones.

    The same rule m10c applies to ``tools/list``: the server is untrusted, so one
    bad entry must not be able to hide the entries that are fine.
    """
    transport = await _transport(tmp_path, "--resources-list", "junk")
    try:
        refs = await list_resources(transport)
    finally:
        await transport.close()
    assert [ref.uri for ref in refs] == [RESOURCE_URI]
    assert refs[0].name == "notes"


async def test_a_resource_with_no_name_or_description_is_still_listed(tmp_path: Path) -> None:
    """Undocumented is not unusable; the URI is what a read needs."""
    transport = await _transport(tmp_path, "--resources-list", "bare")
    try:
        refs = await list_resources(transport)
    finally:
        await transport.close()
    assert refs == (ResourceRef(uri=RESOURCE_URI, name="", description=None, mime_type=None),)


async def test_a_resource_entry_is_not_a_content_block(tmp_path: Path) -> None:
    """The two shapes are different, and the block renderer would mangle this one.

    ``tools/call`` blocks are tagged ``{"type": "text"}``; ``resources/read``
    entries carry ``uri``/``mimeType``/``text`` and no ``type`` at all. Running
    the latter through :func:`~bevel.mcp.bridge._render_blocks` would report every
    real resource as ``[malformed content omitted by bevel]``.
    """
    transport = await _transport(tmp_path)
    try:
        content = await read_resource(transport, RESOURCE_URI)
    finally:
        await transport.close()
    assert content.text == RESOURCE_TEXT
    assert "omitted by bevel" not in content.text


async def test_a_read_of_an_unknown_uri_raises_a_remote_error(tmp_path: Path) -> None:
    """The pre-existing refusal: a JSON-RPC error object, not a silent empty read."""
    transport = await _transport(tmp_path)
    try:
        with pytest.raises(RemoteError) as caught:
            await read_resource(transport, "bevel://fixture/nope")
        assert "bevel://fixture/nope" in str(caught.value)
    finally:
        await transport.close()


async def test_a_listed_resource_the_server_refuses_is_named_by_uri(tmp_path: Path) -> None:
    """The distinct unknown-URI case: the server advertised it, then would not read it.

    This is the one that looks like bevel's bug, because the model got the URI
    from bevel's own list. The refusal has to quote the URI and the server.
    """
    transport = await _transport(tmp_path, "--resources-read", "gone")
    try:
        refs = await list_resources(transport)
        assert GONE_URI in [ref.uri for ref in refs]
        with pytest.raises(RemoteError) as caught:
            await read_resource(transport, GONE_URI)
        assert GONE_URI in str(caught.value)
    finally:
        await transport.close()


async def test_a_read_with_no_contents_is_an_error_not_blank_text(tmp_path: Path) -> None:
    transport = await _transport(tmp_path, "--resources-read", "empty")
    try:
        with pytest.raises(McpHandshakeError) as caught:
            await read_resource(transport, RESOURCE_URI)
        assert RESOURCE_URI in str(caught.value)
    finally:
        await transport.close()


async def test_a_blob_resource_is_described_not_decoded(tmp_path: Path) -> None:
    """Binary is named, not rendered.

    Decoding base64 into the transcript would put mojibake in front of the model
    and call it the resource. bevel says what it is and leaves it alone.
    """
    transport = await _transport(tmp_path, "--resources-read", "blob")
    try:
        content = await read_resource(transport, BLOB_URI)
    finally:
        await transport.close()
    assert content.is_error is False
    assert content.mime_type == "application/octet-stream"
    assert BLOB_SECRET not in content.text
    assert "application/octet-stream" in content.text
    assert "not rendered" in content.text


async def test_an_is_error_read_is_a_failure_not_content(tmp_path: Path) -> None:
    transport = await _transport(tmp_path, "--resources-read", "error")
    try:
        content = await read_resource(transport, RESOURCE_URI)
    finally:
        await transport.close()
    assert content.is_error is True
    assert "the fixture refused to read it" in content.text


async def test_large_content_is_capped(tmp_path: Path) -> None:
    transport = await _transport(tmp_path, "--resources-read", "large")
    try:
        content = await read_resource(transport, RESOURCE_URI)
    finally:
        await transport.close()
    assert content.truncated is True
    assert len(content.text) == MAX_RESOURCE_CHARS
    assert content.text.startswith("abcdefghij")


# ---------------------------------------------------------------------------
# The same wire calls over the second transport.
# ---------------------------------------------------------------------------


async def _start_http(tmp_path: Path, *flags: str) -> tuple[Any, asyncio.subprocess.Process]:
    port_file = tmp_path / "port"
    argv = [sys.executable, str(FIXTURE), "--http", "--port-file", str(port_file), *flags]
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    deadline = asyncio.get_running_loop().time() + 15.0
    while not (port_file.exists() and port_file.read_text(encoding="utf-8").strip()):
        if proc.returncode is not None:
            raise AssertionError(f"fixture exited early with {proc.returncode}")
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the fixture never wrote its port file")
        await asyncio.sleep(0.02)
    url = f"http://127.0.0.1:{port_file.read_text(encoding='utf-8').strip()}"
    return HttpTransport(url, handshake_timeout=HANG_TIMEOUT, call_timeout=HANG_TIMEOUT), proc


async def _stop(transport: Any, proc: asyncio.subprocess.Process) -> None:
    await transport.close()
    if proc.returncode is None:
        proc.kill()
        await proc.wait()


async def test_the_wire_calls_work_over_http(tmp_path: Path) -> None:
    transport, proc = await _start_http(tmp_path)
    try:
        await transport.start()
        refs = await list_resources(transport)
        assert [ref.uri for ref in refs] == [RESOURCE_URI]
        content = await read_resource(transport, RESOURCE_URI)
        assert content.text == RESOURCE_TEXT
    finally:
        await _stop(transport, proc)


async def test_a_hanging_resources_read_over_http_times_out(tmp_path: Path) -> None:
    """One hang implementation, two transports.

    The fixture hangs by emitting nothing, which is how it already reaches HTTP
    mode, so this test would fail if m10f had grown an HTTP-specific hang.
    """
    transport, proc = await _start_http(tmp_path, "--hang", "resources-read")
    try:
        await transport.start()
        with pytest.raises(McpTimeoutError) as caught:
            await read_resource(transport, RESOURCE_URI)
        assert "resources/read" in str(caught.value)
    finally:
        await _stop(transport, proc)


# ---------------------------------------------------------------------------
# The two tools: what the model is actually offered.
# ---------------------------------------------------------------------------


async def test_mcp_resources_lists_uris_names_and_descriptions() -> None:
    servers = await _started()
    try:
        result = await _call(servers, "mcp_resources", server="fixture")
        assert result.ok, result.error
        assert RESOURCE_URI in result.content
        assert "notes" in result.content
        assert "The fixture's one resource." in result.content
        assert "text/plain" in result.content
    finally:
        await servers.aclose()


async def test_mcp_read_returns_the_resource_text() -> None:
    servers = await _started()
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
        assert result.ok, result.error
        assert RESOURCE_TEXT in result.content
        assert RESOURCE_URI in result.content, "the model must be able to tell whose text this is"
        assert "fixture" in result.content
    finally:
        await servers.aclose()


async def test_both_tools_are_read_tier_and_parallel_safe() -> None:
    """PLAN's call: the server list is config the user wrote, not a model-chosen target."""
    servers = await _started()
    try:
        for name in ("mcp_resources", "mcp_read"):
            tool = _tool(servers, name)
            assert tool.risk_tier is RiskTier.READ
            assert tool.is_parallel_safe is True
    finally:
        await servers.aclose()


async def test_the_tool_schemas_are_wire_safe() -> None:
    """Nothing in ``test_tools_registry.py`` looks at these, so this file has to."""
    servers = await _started()
    try:
        for name in ("mcp_resources", "mcp_read"):
            spec = _tool(servers, name).spec
            assert re.fullmatch(r"[a-z][a-z0-9_]*", spec.name)
            assert spec.description.strip()
            assert spec.parameters["type"] == "object"
            assert spec.parameters["additionalProperties"] is False
            assert not unsupported_keywords(spec.parameters)
            for schema in spec.parameters["properties"].values():
                assert str(schema.get("description", "")).strip()
    finally:
        await servers.aclose()


async def test_the_tools_never_enter_the_registry() -> None:
    """A remote server must not be able to change what the static list says."""
    before = registry.specs()
    servers = await _started()
    try:
        assert registry.specs() == before
        assert registry.get("mcp_resources") is None
        assert registry.get("mcp_read") is None
    finally:
        await servers.aclose()


async def test_two_servers_cost_two_tools_not_four() -> None:
    """The reason the tools are global rather than per-server."""
    servers = await start_servers(
        {"one": fixture_config(), "two": fixture_config("--stderr-lines", "1")}
    )
    try:
        offered = [tool.spec.name for tool in resource_tools(servers)]
        assert offered == ["mcp_resources", "mcp_read"]
    finally:
        await servers.aclose()


async def test_no_configured_servers_offers_no_resource_tools() -> None:
    """Two schemas on every request is a real cost for a user who uses no MCP."""
    servers = await start_servers({})
    try:
        assert resource_tools(servers) == ()
    finally:
        await servers.aclose()


async def test_the_tools_are_offered_to_a_run_alongside_the_builtins() -> None:
    provider = MockProvider(turns=[ScriptedTurn(content="done")])
    servers = await _started()
    try:
        agent = Agent(
            provider,
            Path.cwd(),
            tools=[*registry.tools(), *servers.tools, *resource_tools(servers)],
            policy=allow_all_policy,
        )
        result = await agent.run("hello")
        assert result.ok, result.error
        offered = provider.requests[0]["tools"]
        assert isinstance(offered, list)
        assert "mcp_resources" in offered
        assert "mcp_read" in offered
        assert offered[: len(BUILTIN_TOOLS)] == BUILTIN_TOOLS
    finally:
        await servers.aclose()


async def test_a_model_can_list_and_read_in_one_run() -> None:
    """The consumption path, end to end: a fetched resource reaches the transcript."""
    provider = MockProvider(
        turns=[
            ScriptedTurn(
                tool_calls=[
                    tool_call("c1", "mcp_resources", {"server": "fixture"}),
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    tool_call("c2", "mcp_read", {"server": "fixture", "uri": RESOURCE_URI}),
                ]
            ),
            ScriptedTurn(content="done"),
        ]
    )
    servers = await _started()
    try:
        agent = Agent(
            provider,
            Path.cwd(),
            tools=[*registry.tools(), *servers.tools, *resource_tools(servers)],
            policy=allow_all_policy,
        )
        result = await agent.run("read the note")
        assert result.ok, result.error
    finally:
        await servers.aclose()

    sent = provider.requests[-1]["messages"]
    assert isinstance(sent, list)
    observed = [message.content for message in sent if message.role == "tool"]
    assert any(RESOURCE_TEXT in text for text in observed), (
        "the fetched resource never reached the model, which is the whole slice"
    )


# ---------------------------------------------------------------------------
# The four ways a server name fails to reach a transport. Each is a sentence a
# model can act on, and none of them is an empty list.
# ---------------------------------------------------------------------------


async def test_a_name_that_was_never_configured_names_the_ones_that_are() -> None:
    servers = await _started()
    try:
        result = await _call(servers, "mcp_resources", server="typo")
        assert result.failed
        assert "typo" in (result.error or "")
        assert "fixture" in (result.error or ""), "the model cannot recover from a dead end"
    finally:
        await servers.aclose()


async def test_a_disabled_server_says_disabled_rather_than_no_resources() -> None:
    servers = await _started(enabled=False)
    try:
        result = await _call(servers, "mcp_resources", server="fixture")
        assert result.failed
        assert "disabled" in (result.error or "")
        assert "no resources" not in (result.error or "").lower()
    finally:
        await servers.aclose()


async def test_a_server_that_never_handshook_says_so_with_its_error() -> None:
    servers = await start_servers({"broken": fixture_config(command="definitely-not-a-program")})
    try:
        result = await _call(servers, "mcp_resources", server="broken")
        assert result.failed
        error = result.error or ""
        assert "handshook" in error or "never" in error
        assert "definitely-not-a-program" in error, "the report's own reason is the useful part"
    finally:
        await servers.aclose()


async def test_a_lookup_after_aclose_says_closed_and_resurrects_nothing() -> None:
    """A closed transport must not be callable again by way of a tool."""
    servers = await _started()
    await servers.aclose()

    result = await _call(servers, "mcp_resources", server="fixture")
    assert result.failed
    assert "closed" in (result.error or "")
    assert servers.transport("fixture") is None

    result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
    assert result.failed
    assert "closed" in (result.error or "")


async def test_a_closed_server_names_no_available_server() -> None:
    """After the run is over, "servers you can use: fixture" would be a lie."""
    servers = await _started()
    await servers.aclose()
    result = await _call(servers, "mcp_resources", server="fixture")
    assert "fixture" not in (result.error or "").split("Servers you can use:")[-1]


# ---------------------------------------------------------------------------
# What the tools say when the server answers, but answers with nothing useful.
# ---------------------------------------------------------------------------


async def test_a_server_with_no_resources_says_so_rather_than_returning_blank() -> None:
    servers = await _started("--resources-list", "empty")
    try:
        result = await _call(servers, "mcp_resources", server="fixture")
        assert result.ok, result.error
        assert result.content.strip()
        assert "no resources" in result.content.lower()
    finally:
        await servers.aclose()


async def test_a_read_the_server_refuses_is_a_failure_naming_server_and_uri() -> None:
    servers = await _started("--resources-read", "gone")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=GONE_URI)
        assert result.failed
        error = result.error or ""
        assert GONE_URI in error
        assert "fixture" in error
    finally:
        await servers.aclose()


async def test_a_failure_quotes_what_the_server_wrote_to_stderr() -> None:
    """The reuse of ``StartedTransport.stderr_tail``, pinned.

    The bridge quotes it for the same reason: a JSON-RPC error code tells the
    model nothing it can act on, and the server's own last words usually do.
    HTTP answers the same property with its last error body, which is why one
    message works for both transports.
    """
    servers = await _started("--resources-read", "gone", "--stderr-lines", "2")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=GONE_URI)
        assert result.failed
        assert "fixture stderr line 0" in (result.error or "")
    finally:
        await servers.aclose()


async def test_an_is_error_read_is_a_failure_the_model_can_read() -> None:
    servers = await _started("--resources-read", "error")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
        assert result.failed
        assert "the fixture refused to read it" in (result.error or "")
    finally:
        await servers.aclose()


async def test_a_read_with_no_contents_is_a_failure_not_an_empty_success() -> None:
    servers = await _started("--resources-read", "empty")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
        assert result.failed
        assert (result.error or "").strip()
    finally:
        await servers.aclose()


async def test_a_blob_read_says_bevel_did_not_render_it() -> None:
    servers = await _started("--resources-read", "blob")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=BLOB_URI)
        assert result.ok, result.error
        assert BLOB_SECRET not in result.content
        assert "not rendered" in result.content
    finally:
        await servers.aclose()


async def test_a_large_read_is_capped_and_says_it_was_capped() -> None:
    servers = await _started("--resources-read", "large")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
        assert result.ok, result.error
        assert TRUNCATION_MARKER in result.content
        assert str(MAX_RESOURCE_CHARS) in TRUNCATION_MARKER
        assert len(result.content) < MAX_RESOURCE_CHARS + 512
    finally:
        await servers.aclose()


async def test_a_hanging_read_becomes_a_failure_not_a_hung_run() -> None:
    servers = await _started("--hang", "resources-read")
    try:
        result = await _call(servers, "mcp_read", server="fixture", uri=RESOURCE_URI)
        assert result.failed
        assert "resources/read" in (result.error or "")
    finally:
        await servers.aclose()


async def test_a_missing_uri_argument_is_rejected_before_any_call() -> None:
    servers = await _started()
    try:
        tool = _tool(servers, "mcp_read")
        result = await call_tool(tool, {}, ToolContext(workspace=Path.cwd()))
        assert result.failed
        assert "uri" in (result.error or "")
    finally:
        await servers.aclose()

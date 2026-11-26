"""The m10c oracle: bridged tools driven against a real spawned MCP server.

Every test here spawns ``tests/fixtures/mcp_server.py`` as an actual child
process and drives a bridged tool through it. There is no fake transport and no
monkeypatched :class:`~bevel.mcp.stdio.StdioTransport`, on purpose. A bridge
tested against a stubbed transport proves the *mapping* -- that a descriptor
becomes a ``ToolSpec`` -- and proves nothing about whether the call ever reaches
a server. The two things that actually go wrong here, a name the endpoint
rejects at offer time and an ``isError`` that arrives as a result rather than an
exception, only exist on the far side of a real pipe.

The adversarial descriptors live behind the fixture's ``--adversarial-tools``
flag, because m10b pins the default ``tools/list`` to exactly
``["echo", "boom", "methods", "read_note"]``.

Nothing in ``tests/unit/test_tools_registry.py`` looks at these tools -- it
iterates ``registry.tools()``, and bridged tools are deliberately not in the
registry -- so the invariants that file enforces for built-ins (wire-safe name,
``type == "object"``, ``additionalProperties is False``, every property
described, no keywords the validator ignores) are re-checked here instead.

These run in the default lane and are not marked ``slow``: m10d, m10e and m10f
all reuse them.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from bevel.core.agent import Agent
from bevel.core.policy import allow_all_policy
from bevel.core.tools import registry
from bevel.core.tools.base import RiskTier, Tool, ToolContext, call_tool
from bevel.core.tools.schema import unsupported_keywords
from bevel.mcp.bridge import BridgedTool, bridge_tools
from bevel.mcp.stdio import StdioTransport
from bevel.providers.mock import MockProvider, ScriptedTurn

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mcp_server.py"

#: The endpoint's own rule, restated rather than imported from the registry
#: tests: nothing checks a bridged tool for us, so this file has to.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: The nine built-ins, copied from ``test_tools_registry.py`` rather than
#: imported from it, so that someone adding a built-in fails this file loudly
#: instead of quietly making "bridging did not change the registry" vacuous.
BUILTIN_TOOLS = ["read", "grep", "glob", "recall", "todo", "write", "edit", "remember", "shell"]


def fixture_argv(*flags: str) -> list[str]:
    """The exact command line the fixture server is spawned with."""
    return [sys.executable, str(FIXTURE), *flags]


async def _spawn(tmp_path: Path, *flags: str) -> StdioTransport:
    transport = StdioTransport(fixture_argv(*flags), cwd=tmp_path)
    await transport.start()
    return transport


async def _bridged(
    tmp_path: Path, *flags: str, server_name: str = "fixture"
) -> tuple[StdioTransport, Any]:
    """Spawn the fixture and bridge it, handing back both.

    The caller closes the transport. The bridge owns no lifecycle, which is the
    point, so the test has to.
    """
    transport = await _spawn(tmp_path, *flags)
    return transport, await bridge_tools(transport, server_name)


async def _call(tool: BridgedTool, tmp_path: Path, **arguments: Any):
    return await call_tool(tool, arguments, ToolContext(workspace=tmp_path))


# ---------------------------------------------------------------------------
# The round trip. Everything else is a detail of this one.
# ---------------------------------------------------------------------------


async def test_a_bridged_tool_round_trips_through_a_real_server(tmp_path: Path) -> None:
    transport, server = await _bridged(tmp_path)
    try:
        result = await _call(server.by_name("fixture__echo"), tmp_path, text="ping")
        assert result.ok, result.error
        assert "ping" in result.content
    finally:
        await transport.close()


async def test_the_bridge_sends_the_server_s_own_name_not_the_offered_one(
    tmp_path: Path,
) -> None:
    """``fixture__my_tool`` is bevel's name; the wire must carry ``my-tool``.

    If the bridge sent the namespaced name the server would answer "unknown
    tool", and every namespaced tool would break in the one way that looks like
    a bug in somebody else's server.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__my_tool")
        assert tool.remote_name == "my-tool"
        result = await _call(tool, tmp_path, text="hi")
        assert result.ok, result.error
        assert json.loads(result.content) == {"name": "my-tool", "arguments": {"text": "hi"}}
    finally:
        await transport.close()


async def test_arguments_reach_the_server_untouched(tmp_path: Path) -> None:
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        awkward = 'quotes " and a newline\nand a unicode \u00e9'
        result = await _call(server.by_name("fixture__undescribed"), tmp_path, text=awkward)
        assert result.ok, result.error
        assert json.loads(result.content)["arguments"] == {"text": awkward}
    finally:
        await transport.close()


async def test_two_calls_to_the_same_server_both_work(tmp_path: Path) -> None:
    """The bridge must not hold a one-shot handle to the transport."""
    transport, server = await _bridged(tmp_path)
    try:
        echo = server.by_name("fixture__echo")
        assert (await _call(echo, tmp_path, text="one")).content == "one"
        assert (await _call(echo, tmp_path, text="two")).content == "two"
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# Names: namespaced, wire-safe, collision-checked.
# ---------------------------------------------------------------------------


async def test_offered_names_are_namespaced_and_wire_safe(tmp_path: Path) -> None:
    transport, server = await _bridged(tmp_path)
    try:
        assert [tool.spec.name for tool in server.tools] == [
            "fixture__echo",
            "fixture__boom",
            "fixture__methods",
            "fixture__read_note",
        ]
        for tool in server.tools:
            assert NAME_RE.match(tool.spec.name), tool.spec.name
    finally:
        await transport.close()


async def test_a_name_the_endpoint_would_reject_is_sanitised_not_passed_through(
    tmp_path: Path,
) -> None:
    """``Bad Name!`` has to become something an endpoint accepts at offer time.

    One rejected tool name takes the whole offered list down with it, so passing
    this through would cost the model every other tool the server has.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__bad_name")
        assert tool.remote_name == "Bad Name!"
        assert NAME_RE.match(tool.spec.name)
        result = await _call(tool, tmp_path, text="still reachable")
        assert result.ok, result.error
        assert json.loads(result.content)["name"] == "Bad Name!"
    finally:
        await transport.close()


async def test_a_sanitising_collision_refuses_the_second_tool(tmp_path: Path) -> None:
    """``my-tool`` and ``my_tool`` both sanitise to ``my_tool``.

    Keeping one silently makes which tool the model gets depend on the server's
    ordering; renaming the second makes the name on the wire unguessable. The
    second is refused, and the refusal names both originals.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        offered = [tool.spec.name for tool in server.tools]
        assert offered.count("fixture__my_tool") == 1
        assert server.by_name("fixture__my_tool").remote_name == "my-tool"
        refusal = next(r for r in server.refusals if r.remote_name == "my_tool")
        assert "my-tool" in refusal.reason
        assert "my_tool" in refusal.reason
    finally:
        await transport.close()


async def test_two_servers_offering_echo_do_not_collide(tmp_path: Path) -> None:
    """Namespacing is what makes two servers with the same tool usable.

    Both are the same fixture, so their tool lists are identical; only the
    server prefix can tell the model which one it is calling.
    """
    first = await _spawn(tmp_path)
    second = await _spawn(tmp_path)
    try:
        one = await bridge_tools(first, "alpha")
        two = await bridge_tools(second, "beta")
        names = [tool.spec.name for tool in [*one.tools, *two.tools]]
        assert "alpha__echo" in names
        assert "beta__echo" in names
        assert len(names) == len(set(names)), "two servers produced a duplicate offered name"
    finally:
        await first.close()
        await second.close()


async def test_a_server_name_is_sanitised_too(tmp_path: Path) -> None:
    """The prefix is the model's only clue about who runs the code, and in m10d
    it is user-authored config, so it goes through the same rule as the tool."""
    transport, server = await _bridged(tmp_path, server_name="My Server!")
    try:
        assert [tool.spec.name for tool in server.tools][:2] == [
            "my_server__echo",
            "my_server__boom",
        ]
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# Schemas: what gets offered must be something call_tool actually enforces.
# ---------------------------------------------------------------------------


async def test_every_offered_schema_is_one_the_validator_enforces(tmp_path: Path) -> None:
    """The same invariants ``test_tools_registry.py`` holds the built-ins to.
    See the module docstring for why nothing else checks them."""
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        assert server.tools, "the adversarial payload refused everything"
        for tool in server.tools:
            schema = tool.spec.parameters
            assert unsupported_keywords(schema) == set(), tool.spec.name
            assert schema["type"] == "object", tool.spec.name
            assert schema.get("additionalProperties") is False, tool.spec.name
            for name, prop in schema["properties"].items():
                assert prop.get("description"), f"{tool.spec.name}.{name}"
            for name in schema.get("required", []):
                assert name in schema["properties"], f"{tool.spec.name}: {name}"
    finally:
        await transport.close()


async def test_additional_properties_is_forced_false_when_the_server_allows_them(
    tmp_path: Path,
) -> None:
    """A remote schema that permits extras lets a hallucinated argument vanish
    silently: the model invents ``colour``, the server ignores it, and nobody
    ever learns the call was not what was asked for.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__loose_extras")
        assert tool.spec.parameters["additionalProperties"] is False
    finally:
        await transport.close()


async def test_a_hallucinated_extra_argument_never_reaches_the_wire(tmp_path: Path) -> None:
    """``call_tool`` validates against the bridged schema before ``execute``.

    The ``methods`` tool is the proof: it reports what the server actually
    received, so a second ``tools/call`` in that list would mean the invented
    argument went out anyway.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__loose_extras")
        result = await call_tool(tool, {"text": "hi", "colour": "blue"}, ToolContext(tmp_path))
        assert result.failed
        assert "colour" in (result.error or "")

        probe = server.by_name("fixture__methods")
        seen = json.loads((await _call(probe, tmp_path)).content)
        assert seen.count("tools/call") == 1, f"the rejected call reached the server: {seen}"
    finally:
        await transport.close()


async def test_a_tool_with_no_input_schema_still_gets_a_usable_one(tmp_path: Path) -> None:
    """``call_tool`` validates against ``parameters`` whatever it holds, so an
    absent schema becomes a closed empty object rather than ``{}`` -- which the
    validator treats as "anything passes".
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__no_schema")
        assert tool.spec.parameters == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        assert (await _call(tool, tmp_path)).ok
    finally:
        await transport.close()


async def test_an_undocumented_property_is_still_described(tmp_path: Path) -> None:
    """Undocumented parameters produce badly-formed calls from small models, so
    the registry requires a description. The server did not supply one; bevel
    says so plainly rather than inventing a meaning it does not know.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        tool = server.by_name("fixture__undescribed")
        description = tool.spec.parameters["properties"]["text"]["description"]
        assert description
        assert "undocumented" in description.lower()
    finally:
        await transport.close()


@pytest.mark.parametrize(
    ("remote_name", "keyword"),
    [("ref_tool", "$ref"), ("any_of_tool", "anyOf"), ("pattern_tool", "pattern")],
)
async def test_a_schema_bevel_cannot_validate_is_refused(
    tmp_path: Path, remote_name: str, keyword: str
) -> None:
    """Offering it would advertise a constraint bevel does not enforce.

    ``call_tool`` validates against whatever is in ``parameters``, so a ``$ref``
    that survives into the offered schema means the model may send anything and
    be told it was checked.
    """
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        assert all(tool.remote_name != remote_name for tool in server.tools)
        refusal = next(r for r in server.refusals if r.remote_name == remote_name)
        assert keyword in refusal.reason
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# One bad descriptor must not sink the server.
# ---------------------------------------------------------------------------


async def test_one_bad_descriptor_does_not_sink_the_server(tmp_path: Path) -> None:
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        offered = [tool.spec.name for tool in server.tools]
        # The well-formed ones all survive, including those listed *after* the
        # broken ones.
        assert "fixture__echo" in offered
        assert "fixture__bad_name" in offered
        assert "fixture__my_tool" in offered
        assert "fixture__undescribed" in offered
        assert {r.remote_name for r in server.refusals} == {
            "ref_tool",
            "any_of_tool",
            "pattern_tool",
            None,  # the descriptor with no name member
            "blank_description",
            "schema_not_object",
            "my_tool",
        }
    finally:
        await transport.close()


@pytest.mark.parametrize(
    ("remote_name", "phrase"),
    [("blank_description", "description"), ("schema_not_object", "inputSchema")],
)
async def test_a_refusal_says_why_in_words_a_human_can_act_on(
    tmp_path: Path, remote_name: str, phrase: str
) -> None:
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        refusal = next(r for r in server.refusals if r.remote_name == remote_name)
        assert phrase.lower() in refusal.reason.lower()
        assert remote_name in refusal.reason
    finally:
        await transport.close()


async def test_a_descriptor_with_no_name_is_refused_rather_than_named_after_its_description(
    tmp_path: Path,
) -> None:
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        refusal = next(r for r in server.refusals if r.remote_name is None)
        assert "name" in refusal.reason.lower()
    finally:
        await transport.close()


async def test_refusals_are_logged_where_a_human_will_find_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A silently missing tool is the worst outcome: the user cannot tell a
    refused tool from a server that never offered it."""
    with caplog.at_level("WARNING", logger="bevel.mcp.bridge"):
        transport, server = await _bridged(tmp_path, "--adversarial-tools")
        try:
            assert server.refusals, "the adversarial payload refused nothing"
            assert caplog.records, "refusals were not logged"
            assert all(r.levelname == "WARNING" for r in caplog.records)
            text = "\n".join(r.getMessage() for r in caplog.records)
            for refusal in server.refusals:
                assert refusal.remote_name is None or refusal.remote_name in text
            assert "fixture" in text, "the log line does not name the server"
        finally:
            await transport.close()


# ---------------------------------------------------------------------------
# Failures are observations, not exceptions.
# ---------------------------------------------------------------------------


async def test_is_error_comes_back_as_a_failed_result_not_an_exception(tmp_path: Path) -> None:
    """A failed remote call is something the model reads and recovers from.

    Raising here would abort the iteration and cost the user the run, which is
    the wrong response to a server saying "that did not work".
    """
    transport, server = await _bridged(tmp_path)
    try:
        result = await _call(server.by_name("fixture__boom"), tmp_path)
        assert result.failed
        assert not result.ok
        assert "boom" in (result.error or "")
    finally:
        await transport.close()


async def test_a_failed_call_names_the_server_the_remote_tool_and_the_stderr_tail(
    tmp_path: Path,
) -> None:
    transport, server = await _bridged(tmp_path, "--stderr-lines", "2")
    try:
        result = await _call(server.by_name("fixture__boom"), tmp_path)
        error = result.error or ""
        assert "fixture" in error, "the server is not named"
        assert "boom" in error, "the remote tool is not named"
        assert "fixture stderr line 0" in error, "the stderr tail is not quoted"
    finally:
        await transport.close()


async def test_a_server_side_is_error_flag_surfaces_as_a_failure(tmp_path: Path) -> None:
    """``--is-error`` makes *every* call fail, including a well-formed echo."""
    transport, server = await _bridged(tmp_path, "--is-error")
    try:
        result = await _call(server.by_name("fixture__echo"), tmp_path, text="hi")
        assert result.failed
        assert "fixture" in (result.error or "")
    finally:
        await transport.close()


async def test_a_transport_failure_surfaces_as_a_failed_result_not_an_exception(
    tmp_path: Path,
) -> None:
    """The server dies mid-session; the model gets an observation, not a crash."""
    transport = await _spawn(tmp_path)
    server = await bridge_tools(transport, "fixture")
    try:
        await transport.kill()
        result = await _call(server.by_name("fixture__echo"), tmp_path, text="hi")
        assert result.failed
        assert "fixture" in (result.error or "")
        assert "echo" in (result.error or "")
    finally:
        await transport.close()


# ---------------------------------------------------------------------------
# The tier, and the boundary with the registry.
# ---------------------------------------------------------------------------


async def test_every_bridged_tool_is_exec_unsafe_and_serial(tmp_path: Path) -> None:
    """The server is third-party code and the ladder says a tool cannot promote
    itself, so there is no per-tool override and no descriptor field that can
    change this."""
    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        assert server.tools
        for tool in server.tools:
            assert tool.risk_tier is RiskTier.EXEC_UNSAFE, tool.spec.name
            assert tool.is_parallel_safe is False, tool.spec.name
    finally:
        await transport.close()


async def test_bridged_tools_satisfy_the_tool_protocol(tmp_path: Path) -> None:
    transport, server = await _bridged(tmp_path)
    try:
        for tool in server.tools:
            assert isinstance(tool, Tool), tool.spec.name
    finally:
        await transport.close()


async def test_bridging_does_not_touch_the_registry(tmp_path: Path) -> None:
    """The registry is the source of truth for *built-in* tools.

    A remote tool appearing there would make the statically offered list depend
    on whatever the user happens to have configured, so two users with the same
    bevel would be offered different tools.
    """
    before = registry.specs()
    assert [spec.name for spec in before] == BUILTIN_TOOLS

    transport, server = await _bridged(tmp_path, "--adversarial-tools")
    try:
        assert server.tools, "nothing was bridged, so this assertion would be vacuous"
        assert registry.specs() == before
        assert [spec.name for spec in registry.specs()] == BUILTIN_TOOLS
        assert registry.get("fixture__echo") is None
    finally:
        await transport.close()


async def test_bridged_tools_reach_the_model_only_through_the_agent(tmp_path: Path) -> None:
    provider = MockProvider(turns=[ScriptedTurn(content="done")])
    transport, server = await _bridged(tmp_path)
    try:
        agent = Agent(
            provider,
            tmp_path,
            tools=[*registry.tools(), *server.tools],
            policy=allow_all_policy,
        )
        result = await agent.run("hello")
        assert result.ok, result.error
        offered = provider.requests[0]["tools"]
        assert isinstance(offered, list)
        assert "fixture__echo" in offered
        assert offered[: len(BUILTIN_TOOLS)] == BUILTIN_TOOLS
    finally:
        await transport.close()

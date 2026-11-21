"""The orient step: the project map rides on the first request, unasked.

Two promises are checked here. The first is the one the user feels: a greeting
comes back with zero tool calls, because the orientation the model would have
spent three calls discovering is already in its context. The second is that the
map actually reached the wire -- ``MockProvider.requests`` is the record of what
was sent, which is the only way to tell "injected" apart from "implemented and
never called".
"""

from __future__ import annotations

from pathlib import Path

from bevel.core.agent import Agent
from bevel.core.policy import allow_all_policy
from bevel.core.tools._fs import SKIP_DIRS
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call


def _agent(workspace: Path, *turns: ScriptedTurn) -> Agent:
    return Agent(
        MockProvider(turns=list(turns)),
        workspace,
        policy=allow_all_policy,
        tools=[],
    )


async def test_greeting_completes_with_zero_tool_calls(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    agent = _agent(tmp_path, ScriptedTurn(content="Hi. What do you need?"))
    result = await agent.run("hello")

    assert result.tool_calls == 0
    assert result.iterations == 1
    assert result.ok
    assert result.final_text.strip(), "a greeting must be answered with words"


async def test_project_map_is_on_the_wire_for_the_first_request(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n\nThe intro.\n", encoding="utf-8")
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    result = await Agent(provider, tmp_path, policy=allow_all_policy, tools=[]).run("hello")

    sent = provider.requests[0]["messages"]
    assert isinstance(sent, list)
    text = "\n".join(message.content for message in sent)
    assert "# project map:" in text
    assert "## pyproject.toml" in text
    assert "The intro." in text
    assert result.tool_calls == 0


async def test_map_appears_once_in_the_transcript_however_long_the_run(tmp_path: Path) -> None:
    """Two iterations must not mean two maps.

    The call is to an unregistered tool, which the loop answers with an
    ``unavailable`` observation and iterates again -- a second round trip with no
    filesystem involved.
    """
    provider = MockProvider(
        turns=[
            ScriptedTurn(content="", tool_calls=(tool_call("c1", "teleport", {}),)),
            ScriptedTurn(content="done"),
        ]
    )
    result = await Agent(provider, tmp_path, policy=allow_all_policy, tools=[]).run("go")

    assert result.iterations == 2
    maps = [m for m in result.messages if "# project map:" in m.content]
    assert len(maps) == 1, "the map was injected once per iteration"


async def test_junk_directories_never_reach_the_model(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "site-packages").mkdir(parents=True)
    (tmp_path / "src").mkdir()
    provider = MockProvider(turns=[ScriptedTurn(content="hi")])
    await Agent(provider, tmp_path, policy=allow_all_policy, tools=[]).run("hello")

    sent = provider.requests[0]["messages"]
    assert isinstance(sent, list)
    text = "\n".join(message.content for message in sent)
    junk = next(iter(SKIP_DIRS))
    assert junk not in text
    assert "src/" in text


async def test_unreadable_workspace_does_not_stop_the_greeting(tmp_path: Path) -> None:
    """A workspace that cannot be walked must not cost the user their reply."""
    missing = tmp_path / "gone"
    provider = MockProvider(turns=[ScriptedTurn(content="Hello!")])
    result = await Agent(provider, missing, policy=allow_all_policy, tools=[]).run("hello")

    assert result.ok
    assert result.final_text == "Hello!"
    sent = provider.requests[0]["messages"]
    assert isinstance(sent, list)
    assert any("does not exist" in message.content for message in sent)

"""Tests for the tool registry.

``core/tools/types.py`` names this module as the single source of truth for the
tool list the model is offered, so these tests hold it to that: the six m03
tools are all present, nothing is offered twice, and every declared schema is
one the hand-rolled validator actually understands -- a typo in a keyword would
otherwise silently disable argument checking.
"""

from __future__ import annotations

import re
from pathlib import Path

from bevel.core.tools import registry
from bevel.core.tools.base import RiskTier, Tool, ToolContext, call_tool
from bevel.core.tools.schema import unsupported_keywords

M03_TOOLS = ["read", "grep", "glob", "write", "edit", "shell"]
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class TestToolList:
    def test_the_six_m03_tools_are_offered(self) -> None:
        assert [tool.spec.name for tool in registry.tools()] == M03_TOOLS

    def test_the_list_is_stable_across_calls(self) -> None:
        assert [tool.spec.name for tool in registry.tools()] == [
            tool.spec.name for tool in registry.tools()
        ]

    def test_names_are_unique(self) -> None:
        names = [tool.spec.name for tool in registry.tools()]
        assert len(names) == len(set(names))

    def test_names_are_wire_safe(self) -> None:
        """OpenAI-compatible endpoints reject tool names outside [a-zA-Z0-9_-]."""
        for tool in registry.tools():
            assert NAME_RE.match(tool.spec.name), tool.spec.name

    def test_every_tool_has_a_description_the_model_can_act_on(self) -> None:
        for tool in registry.tools():
            assert len(tool.spec.description) > 20, tool.spec.name
            assert tool.spec.description[0].isupper()

    def test_every_tool_declares_a_risk_tier(self) -> None:
        for tool in registry.tools():
            assert isinstance(tool.risk_tier, RiskTier), tool.spec.name

    def test_only_read_only_tools_are_parallel_safe(self) -> None:
        for tool in registry.tools():
            if tool.is_parallel_safe:
                assert tool.risk_tier is RiskTier.READ, tool.spec.name

    def test_no_tier_without_an_m03_producer_is_declared(self) -> None:
        """m05 and m12 own the other tiers; declaring them here would be a lie."""
        declared = {tool.risk_tier for tool in registry.tools()}
        assert declared == {RiskTier.READ, RiskTier.WRITE_WORKSPACE, RiskTier.EXEC_UNSAFE}


class TestSchemas:
    def test_every_schema_uses_only_keywords_the_validator_understands(self) -> None:
        for tool in registry.tools():
            assert unsupported_keywords(tool.spec.parameters) == set(), tool.spec.name

    def test_every_schema_is_an_object_with_properties(self) -> None:
        for tool in registry.tools():
            schema = tool.spec.parameters
            assert schema["type"] == "object", tool.spec.name
            assert schema["properties"], tool.spec.name

    def test_required_names_are_all_declared_properties(self) -> None:
        for tool in registry.tools():
            schema = tool.spec.parameters
            for name in schema.get("required", []):
                assert name in schema["properties"], f"{tool.spec.name}: {name}"

    def test_every_property_is_described(self) -> None:
        """Undocumented parameters produce badly-formed calls from small models."""
        for tool in registry.tools():
            for name, schema in tool.spec.parameters["properties"].items():
                assert schema.get("description"), f"{tool.spec.name}.{name}"

    def test_additional_properties_are_rejected(self) -> None:
        """Hallucinated arguments should fail loudly instead of being ignored."""
        for tool in registry.tools():
            assert tool.spec.parameters.get("additionalProperties") is False, tool.spec.name


class TestLookup:
    def test_lookup_by_name(self) -> None:
        assert registry.get("read").spec.name == "read"

    def test_unknown_names_return_none(self) -> None:
        assert registry.get("apply_patch") is None

    def test_specs_are_the_tool_specs(self) -> None:
        assert registry.specs() == [tool.spec for tool in registry.tools()]

    async def test_every_tool_can_be_constructed_and_executed_against_a_context(
        self, tmp_path: object
    ) -> None:
        """Guards against a registry entry that is a class, an instance, or neither."""
        ctx = ToolContext(workspace=Path(str(tmp_path)))
        for tool in registry.tools():
            assert isinstance(tool, Tool), tool.spec.name
            # Deliberately invalid arguments: the seam must reject them, which
            # proves every entry really is a Tool with a working execute().
            result = await call_tool(tool, {}, ctx)
            assert result.failed, tool.spec.name


class TestModuleShape:
    def test_the_registry_exports_the_names_types_py_promise(self) -> None:
        for name in ("tools", "get", "specs"):
            assert hasattr(registry, name), name

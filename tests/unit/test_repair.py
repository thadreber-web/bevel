"""Tests for bounded, observable tool-call repair.

The instruction file's rule is the centre of gravity here: *repairing syntax is
fine, inventing an argument is fabrication*. Most of these tests therefore
assert not just what repair can do but what it must refuse to do.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from bevel.core.messages import Message, ToolCall
from bevel.core.tools.schema import SchemaError
from bevel.core.tools.types import ToolSpec
from bevel.providers.base import AssistantTurn, Capabilities, TransportError
from bevel.providers.mock import MockProvider, ScriptedTurn
from bevel.providers.repair import (
    DEFAULT_MAX_ATTEMPTS,
    RepairOutcome,
    ToolCallRepairer,
    extract_json_object,
    format_violations,
)

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "append": {"type": "boolean"},
    },
    "required": ["path", "content"],
}

WRITE = ToolSpec(name="write", description="Write a file", parameters=WRITE_SCHEMA)


def call(arguments: str, *, name: str = "write", call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def provider_with(*replies: str, **kwargs: Any) -> MockProvider:
    return MockProvider(turns=[ScriptedTurn(content=text) for text in replies], **kwargs)


# -- extract_json_object ----------------------------------------------------


class TestExtractJSONObject:
    def test_clean_object_needs_no_fixes(self) -> None:
        parsed, fixes = extract_json_object('{"path": "a.py", "content": "x"}')
        assert parsed == {"path": "a.py", "content": "x"}
        assert fixes == ()

    def test_surrounding_whitespace_is_named(self) -> None:
        parsed, fixes = extract_json_object('\n  {"a": 1}  \n')
        assert parsed == {"a": 1}
        assert fixes == ("trimmed whitespace",)

    def test_code_fence_with_language_tag(self) -> None:
        parsed, fixes = extract_json_object('```json\n{"a": 1}\n```')
        assert parsed == {"a": 1}
        assert "stripped code fence" in fixes

    def test_code_fence_without_language_tag(self) -> None:
        parsed, fixes = extract_json_object('```\n{"a": 1}\n```')
        assert parsed == {"a": 1}
        assert "stripped code fence" in fixes

    def test_prose_wrapped_object_is_extracted(self) -> None:
        """Models routinely answer "sure, here you go: {...}"."""
        parsed, fixes = extract_json_object('Sure. Here you go: {"a": 1} -- anything else?')
        assert parsed == {"a": 1}
        assert "extracted outermost object" in fixes

    def test_trailing_comma_in_object(self) -> None:
        parsed, fixes = extract_json_object('{"a": 1,}')
        assert parsed == {"a": 1}
        assert "dropped trailing commas" in fixes

    def test_trailing_comma_in_array(self) -> None:
        parsed, fixes = extract_json_object('{"a": [1, 2,]}')
        assert parsed == {"a": [1, 2]}
        assert "dropped trailing commas" in fixes

    def test_trailing_comma_inside_prose(self) -> None:
        parsed, fixes = extract_json_object('result: {"a": [1,],}')
        assert parsed == {"a": [1]}
        assert set(fixes) == {"extracted outermost object", "dropped trailing commas"}

    def test_a_comma_inside_a_string_value_is_left_alone(self) -> None:
        """The naive fix must not corrupt data. ",}" inside a string is content."""
        parsed, _fixes = extract_json_object('{"content": "end,}"}')
        assert parsed == {"content": "end,}"}

    def test_unparseable_returns_none_with_the_fixes_tried(self) -> None:
        parsed, fixes = extract_json_object("I could not do that.")
        assert parsed is None
        assert fixes == ()

    def test_json_that_is_not_an_object_is_rejected(self) -> None:
        """A bare array or string is not arguments, even though it parses."""
        assert extract_json_object("[1, 2]")[0] is None
        assert extract_json_object('"hello"')[0] is None
        assert extract_json_object("42")[0] is None
        assert extract_json_object("null")[0] is None

    def test_empty_and_blank_input(self) -> None:
        assert extract_json_object("")[0] is None
        assert extract_json_object("   ")[0] is None

    def test_nested_objects_survive_intact(self) -> None:
        text = '{"a": {"b": {"c": [1, {"d": 2}]}}}'
        parsed, fixes = extract_json_object(text)
        assert parsed == {"a": {"b": {"c": [1, {"d": 2}]}}}
        assert fixes == ()

    def test_unbalanced_braces_in_prose_do_not_crash(self) -> None:
        assert extract_json_object("here { you go")[0] is None


# -- format_violations ------------------------------------------------------


class TestFormatViolations:
    def test_joins_every_violation(self) -> None:
        """All problems go back in one re-ask; one-at-a-time wastes attempts."""
        text = format_violations(
            [SchemaError("$.path", "is required"), SchemaError("$.n", "not an int")]
        )
        assert text == "$.path: is required; $.n: not an int"

    def test_empty_sequence_is_empty_string(self) -> None:
        assert format_violations([]) == ""


# -- repair: no re-ask needed ------------------------------------------------


class TestRepairWithoutReAsk:
    async def test_valid_arguments_pass_through_unchanged(self) -> None:
        provider = provider_with("SHOULD NOT BE ASKED")
        outcome = await ToolCallRepairer(provider).repair(
            call('{"path": "a.py", "content": "x"}'), WRITE
        )
        assert outcome.ok is True
        assert outcome.arguments == {"path": "a.py", "content": "x"}
        assert outcome.attempts == 0
        assert outcome.repaired is False
        assert provider.requests == []

    async def test_arguments_are_rewritten_from_the_parsed_form(self) -> None:
        provider = provider_with()
        outcome = await ToolCallRepairer(provider).repair(
            call('{"path": "a",  "content":"b"}'), WRITE
        )
        assert outcome.call is not None
        assert json.loads(outcome.call.arguments) == {"path": "a", "content": "b"}
        assert outcome.call.id == "call_1"
        assert outcome.call.name == "write"

    async def test_syntax_repair_is_reported_not_hidden(self) -> None:
        provider = provider_with()
        outcome = await ToolCallRepairer(provider).repair(
            call('```json\n{"path": "a", "content": "b",}\n```'), WRITE
        )
        assert outcome.ok is True
        assert outcome.repaired is True
        assert "stripped code fence" in outcome.fixes
        assert "dropped trailing commas" in outcome.fixes

    async def test_repair_never_adds_a_missing_required_property(self) -> None:
        """The hard rule: if the model did not say it, bevel does not say it."""
        provider = provider_with()  # no scripted reply -> re-asks come back empty
        outcome = await ToolCallRepairer(provider).repair(call('{"path": "a.py"}'), WRITE)
        assert outcome.ok is False
        assert outcome.arguments is None

    async def test_repair_never_coerces_a_type(self) -> None:
        provider = provider_with()
        outcome = await ToolCallRepairer(provider).repair(
            call('{"path": "a.py", "content": "x", "append": "yes"}'), WRITE
        )
        assert outcome.ok is False
        assert any("append" in error for error in outcome.errors)


# -- repair: re-asking -------------------------------------------------------


class TestRepairWithReAsk:
    async def test_re_ask_receives_the_violations_and_the_schema(self) -> None:
        provider = provider_with('{"path": "a.py", "content": "fixed"}')
        outcome = await ToolCallRepairer(provider).repair(call('{"path": "a.py"}'), WRITE)
        assert outcome.ok is True
        assert outcome.arguments == {"path": "a.py", "content": "fixed"}
        assert outcome.attempts == 1
        assert any(fix == "re-asked attempt 1" for fix in outcome.fixes)

        prompt = str(provider.requests[0]["messages"][-1].content)
        assert "`write`" in prompt
        assert "Expected argument schema" in prompt
        assert "Do not invent values" in prompt
        assert '{"path": "a.py"}' in prompt

    async def test_history_is_included_so_the_re_ask_is_answerable(self) -> None:
        provider = provider_with('{"path": "a.py", "content": "x"}')
        history = [Message.user("read it"), Message("assistant", content="here it is")]
        await ToolCallRepairer(provider).repair(call('{"path": "a.py"}'), WRITE, history=history)
        sent = provider.requests[0]["messages"]
        assert [m.role for m in sent] == ["user", "assistant", "user"]

    async def test_json_mode_only_when_the_endpoint_supports_it(self) -> None:
        capable = provider_with(
            '{"path": "a", "content": "b"}',
            capabilities=Capabilities(supports_structured_output=True),
        )
        await ToolCallRepairer(capable).repair(call('{"path": "a"}'), WRITE)
        options = capable.requests[0]["options"]
        assert options is not None
        assert options.json_mode is True
        assert options.temperature == 0.0

        plain = provider_with('{"path": "a", "content": "b"}')
        await ToolCallRepairer(plain).repair(call('{"path": "a"}'), WRITE)
        options = plain.requests[0]["options"]
        assert options is None

    async def test_re_ask_never_asks_for_tools(self) -> None:
        provider = provider_with('{"path": "a", "content": "b"}')
        await ToolCallRepairer(provider).repair(call('{"path": "a"}'), WRITE)
        assert provider.requests[0]["tools"] == []

    async def test_second_attempt_can_succeed(self) -> None:
        provider = provider_with("not json at all", '{"path": "a", "content": "b"}')
        outcome = await ToolCallRepairer(provider).repair(call('{"path": "a"}'), WRITE)
        assert outcome.ok is True
        assert outcome.attempts == 2
        assert len(provider.requests) == 2

    async def test_attempts_are_bounded(self) -> None:
        provider = provider_with("nope", "still nope", "never")
        outcome = await ToolCallRepairer(provider, max_attempts=2).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert outcome.attempts == 2
        assert len(provider.requests) == 2

    async def test_default_max_attempts_is_small(self) -> None:
        """A model that cannot produce JSON should fail fast, not eat context."""
        assert DEFAULT_MAX_ATTEMPTS == 2

    async def test_zero_attempts_never_contacts_the_model(self) -> None:
        provider = provider_with('{"path": "a", "content": "b"}')
        outcome = await ToolCallRepairer(provider, max_attempts=0).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert outcome.attempts == 0
        assert provider.requests == []

    async def test_negative_attempts_is_a_configuration_error(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ToolCallRepairer(provider_with(), max_attempts=-1)

    async def test_errors_describe_why_it_gave_up(self) -> None:
        provider = provider_with('{"path": "a"}')
        outcome = await ToolCallRepairer(provider, max_attempts=1).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert any("content" in error for error in outcome.errors)

    async def test_unparseable_re_ask_is_reported_as_such(self) -> None:
        provider = provider_with("I refuse.")
        outcome = await ToolCallRepairer(provider, max_attempts=1).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert any("not a JSON object" in error for error in outcome.errors)

    async def test_non_object_re_ask_is_rejected(self) -> None:
        provider = provider_with("[1, 2, 3]")
        outcome = await ToolCallRepairer(provider, max_attempts=1).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False


# -- repair: failure modes ---------------------------------------------------


class TestRepairFailureModes:
    async def test_transport_failure_stops_immediately(self) -> None:
        """If the server is unreachable, a second attempt fails the same way."""

        class Down(MockProvider):
            async def complete(self, *_args: Any, **_kwargs: Any) -> AssistantTurn:
                raise TransportError("connection refused")

        provider = Down(turns=[])
        outcome = await ToolCallRepairer(provider, max_attempts=3).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert outcome.attempts == 1
        assert provider.requests == []

    async def test_empty_reply_counts_as_an_attempt_but_yields_nothing(self) -> None:
        provider = MockProvider(turns=[ScriptedTurn(content="")])
        outcome = await ToolCallRepairer(provider, max_attempts=1).repair(
            call('{"path": "a"}'), WRITE
        )
        assert outcome.ok is False
        assert outcome.attempts == 1

    async def test_garbage_arguments_are_reported_not_raised(self) -> None:
        provider = provider_with()
        outcome = await ToolCallRepairer(provider, max_attempts=0).repair(call("<>>"), WRITE)
        assert outcome.ok is False
        assert any("not a JSON object" in error for error in outcome.errors)

    async def test_unknown_tool_schema_without_properties_accepts_any_object(self) -> None:
        spec = ToolSpec(name="ping", description="Ping", parameters={"type": "object"})
        provider = provider_with()
        outcome = await ToolCallRepairer(provider).repair(call("{}"), spec)
        assert outcome.ok is True
        assert outcome.arguments == {}


class TestRepairOutcome:
    def test_ok_tracks_arguments(self) -> None:
        assert RepairOutcome(call=None, arguments=None).ok is False
        assert RepairOutcome(call=None, arguments={}).ok is True

    def test_repaired_requires_both_arguments_and_fixes(self) -> None:
        assert RepairOutcome(call=None, arguments={"a": 1}).repaired is False
        assert (
            RepairOutcome(call=None, arguments={"a": 1}, fixes=("trimmed whitespace",)).repaired
            is True
        )
        assert (
            RepairOutcome(call=None, arguments=None, fixes=("re-asked attempt 1",)).repaired
            is False
        )

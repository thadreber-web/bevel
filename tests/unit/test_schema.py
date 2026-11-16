"""Tests for the hand-rolled JSON Schema validator.

This validator stands in for ``jsonschema`` under the three-runtime-dependency
rule, so the tests have to cover two things: that it rejects what the real
validator would reject, and that it does not silently ignore a constraint it
was never asked to implement.
"""

from __future__ import annotations

import pytest

from bevel.core.tools.schema import SchemaError, is_valid, unsupported_keywords, validate

WRITE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "content": {"type": "string"},
        "append": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["create", "overwrite"]},
        "retries": {"type": "integer", "minimum": 0, "maximum": 5},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}


def paths(errors: list[SchemaError]) -> list[str]:
    return [error.path for error in errors]


class TestHappyPath:
    def test_valid_arguments_produce_no_errors(self) -> None:
        args = {"path": "a.py", "content": "x", "mode": "create", "retries": 2, "tags": ["a"]}
        assert validate(args, WRITE_SCHEMA) == []
        assert is_valid(args, WRITE_SCHEMA) is True

    def test_optional_properties_may_be_omitted(self) -> None:
        assert is_valid({"path": "a", "content": ""}, WRITE_SCHEMA) is True


class TestRequiredAndTypes:
    def test_missing_required_property_is_reported_at_its_own_path(self) -> None:
        errors = validate({"path": "a"}, WRITE_SCHEMA)
        assert paths(errors) == ["$.content"]
        assert "required" in str(errors[0])

    def test_wrong_type_reports_what_it_got(self) -> None:
        errors = validate({"path": 5, "content": ""}, WRITE_SCHEMA)
        assert paths(errors) == ["$.path"]
        assert "got integer" in str(errors[0])

    def test_all_violations_come_back_in_one_pass(self) -> None:
        """One re-ask should fix everything, so the validator must not stop early."""
        errors = validate({"path": "", "content": 1, "mode": "delete", "retries": 99}, WRITE_SCHEMA)
        assert set(paths(errors)) == {"$.path", "$.content", "$.mode", "$.retries"}

    def test_bool_is_not_an_integer(self) -> None:
        """Python's bool subclasses int; JSON Schema does not allow the swap."""
        assert is_valid({"path": "a", "content": "", "retries": True}, WRITE_SCHEMA) is False

    def test_int_is_a_number_but_bool_is_not(self) -> None:
        assert is_valid(1, {"type": "number"}) is True
        assert is_valid(True, {"type": "number"}) is False
        assert is_valid(True, {"type": "boolean"}) is True
        assert is_valid(1, {"type": "boolean"}) is False

    def test_union_types(self) -> None:
        schema = {"type": ["string", "null"]}
        assert is_valid("x", schema) is True
        assert is_valid(None, schema) is True
        assert is_valid(3, schema) is False

    def test_unknown_type_name_does_not_reject(self) -> None:
        """An unrecognised type name is not a constraint we can enforce."""
        assert is_valid("anything", {"type": "timestamp"}) is True


class TestObject:
    def test_additional_properties_false_rejects_extras(self) -> None:
        errors = validate({"path": "a", "content": "", "extra": 1}, WRITE_SCHEMA)
        assert paths(errors) == ["$.extra"]

    def test_additional_properties_schema_validates_extras(self) -> None:
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "additionalProperties": {"type": "integer"},
        }
        assert is_valid({"a": "x", "b": 1}, schema) is True
        assert is_valid({"a": "x", "b": "no"}, schema) is False

    def test_nested_object_paths(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "options": {
                    "type": "object",
                    "properties": {"timeout": {"type": "integer", "minimum": 1}},
                    "required": ["timeout"],
                }
            },
            "required": ["options"],
        }
        errors = validate({"options": {"timeout": 0}}, schema)
        assert paths(errors) == ["$.options.timeout"]

    def test_empty_schema_accepts_anything(self) -> None:
        assert is_valid({"anything": [1, 2]}, {}) is True
        assert is_valid(5, {}) is True

    def test_false_schema_rejects(self) -> None:
        assert is_valid({"a": 1}, {"properties": {"a": False}}) is False

    def test_malformed_schema_fragment_is_an_error_not_a_crash(self) -> None:
        errors = validate({"a": 1}, {"properties": {"a": "not-a-schema"}})
        assert paths(errors) == ["$.a"]
        assert "malformed" in str(errors[0])


class TestArray:
    def test_items_validated_with_indexed_paths(self) -> None:
        errors = validate({"path": "a", "content": "", "tags": ["ok", 7]}, WRITE_SCHEMA)
        assert paths(errors) == ["$.tags[1]"]

    def test_max_items(self) -> None:
        errors = validate({"path": "a", "content": "", "tags": ["1", "2", "3", "4"]}, WRITE_SCHEMA)
        assert any("at most 3" in str(e) for e in errors)

    def test_min_items(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 2}
        assert is_valid(["a"], schema) is False
        assert is_valid(["a", "b"], schema) is True


class TestNumberAndString:
    @pytest.mark.parametrize(
        ("value", "valid"),
        [(0, True), (5, True), (-1, False), (6, False)],
    )
    def test_inclusive_bounds(self, value: int, valid: bool) -> None:
        assert is_valid(value, {"type": "integer", "minimum": 0, "maximum": 5}) is valid

    def test_exclusive_bounds(self) -> None:
        schema = {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 5}
        assert is_valid(1, schema) is True
        assert is_valid(0, schema) is False
        assert is_valid(5, schema) is False

    def test_string_length_bounds(self) -> None:
        schema = {"type": "string", "minLength": 2, "maxLength": 4}
        assert is_valid("ab", schema) is True
        assert is_valid("a", schema) is False
        assert is_valid("abcde", schema) is False

    def test_enum(self) -> None:
        errors = validate("delete", {"type": "string", "enum": ["create", "overwrite"]})
        assert "not one of" in str(errors[0])


class TestUnsupportedKeywords:
    def test_supported_keywords_are_not_reported(self) -> None:
        assert unsupported_keywords(WRITE_SCHEMA) == set()

    def test_property_names_are_not_reported_as_keywords(self) -> None:
        """A parameter called ``type`` or ``items`` is not an unsupported keyword."""
        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
                "enum": {"type": "string"},
                "default": {"type": "string"},
            },
            "required": ["type"],
        }
        assert unsupported_keywords(schema) == set()

    def test_pattern_properties_keys_are_regexes_not_keywords(self) -> None:
        schema = {"type": "object", "patternProperties": {"^x_": {"type": "string"}}}
        assert unsupported_keywords(schema) == {"patternProperties"}

    def test_enum_values_are_not_walked_as_schemas(self) -> None:
        # An enum value that looks like a schema must not be mistaken for one.
        schema = {"type": "object", "properties": {"mode": {"enum": [{"type": "object"}]}}}
        assert unsupported_keywords(schema) == set()

    def test_unenforced_keyword_is_discovered(self) -> None:
        """The whole point: a tool author learns about this now, not in production."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string", "pattern": "^a$"}},
            "anyOf": [{"type": "string"}],
        }
        assert unsupported_keywords(schema) == {"pattern", "anyOf"}

    def test_unsupported_keyword_does_not_change_validation(self) -> None:
        """Documenting the gap honestly: an unenforced constraint lets values through."""
        schema = {"type": "string", "pattern": "^a$"}
        assert is_valid("zzz", schema) is True

    def test_metadata_keywords_are_supported(self) -> None:
        schema = {"type": "string", "description": "d", "title": "t", "default": "x"}
        assert unsupported_keywords(schema) == set()

    def test_descends_into_lists(self) -> None:
        assert unsupported_keywords({"items": {"format": "date"}}) == {"format"}

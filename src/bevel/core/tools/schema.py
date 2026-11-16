"""A deliberately small JSON Schema validator.

``jsonschema`` is not a declared dependency and adding one to check a handful
of tool arguments is not worth it. This covers the subset bevel puts in its own
tool declarations -- ``type``, ``properties``, ``required``, ``items``,
``enum``, ``additionalProperties``, ``minimum``/``maximum``,
``minLength``/``maxLength`` -- and nothing else.

The rule for extending it: add a keyword only when a real tool declaration uses
it. An unimplemented keyword is a bug, not a feature, so unknown keywords are
reported rather than quietly ignored -- silently skipping a constraint would
let an invalid argument through with a clean bill of health.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}

_SUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "additionalProperties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "description",
        "default",
        "title",
    }
)

# Keywords whose *value* is a mapping of subschemas, so the keys of that
# mapping are instance property names (or regexes) rather than keywords.
_SUBSCHEMA_MAP_KEYWORDS: frozenset[str] = frozenset({"properties", "patternProperties"})

# Keywords whose value is arbitrary data, not a schema, and must not be walked.
_DATA_KEYWORDS: frozenset[str] = frozenset({"enum", "default", "const"})


@dataclass(frozen=True)
class SchemaError:
    """One violation, with a path like ``arguments.path`` for the message."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def validate(instance: Any, schema: dict[str, Any], *, path: str = "$") -> list[SchemaError]:
    """Validate ``instance`` against ``schema``.

    Returns every violation rather than the first: when the model is asked to
    fix its output, listing all the problems in one round is both cheaper and
    more likely to succeed than fixing them one re-ask at a time.
    """
    errors: list[SchemaError] = []
    _validate_into(instance, schema, path, errors)
    return errors


def is_valid(instance: Any, schema: dict[str, Any]) -> bool:
    return not validate(instance, schema)


def unsupported_keywords(schema: dict[str, Any]) -> set[str]:
    """Keywords in ``schema`` that this validator does not enforce.

    Tool authors call this so an unsupported constraint is discovered while
    writing the tool, not when a bad argument slips through.
    """
    found: set[str] = set()
    _collect(schema, found)
    return found


def _collect(node: Any, found: set[str], *, keys_are_names: bool = False) -> None:
    """Walk a schema, collecting keywords we do not enforce.

    ``keys_are_names`` is set for ``properties``/``patternProperties``, whose
    keys are instance property names (or regexes), not schema keywords.
    Without it every parameter name would be reported as an unsupported
    keyword.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if not keys_are_names and key not in _SUPPORTED_KEYWORDS:
                found.add(key)
            if key in _DATA_KEYWORDS:
                continue
            _collect(value, found, keys_are_names=key in _SUBSCHEMA_MAP_KEYWORDS)
    elif isinstance(node, list):
        for item in node:
            _collect(item, found)


# Ordered: ``bool`` is a subclass of ``int``, so it has to be tested first.
_PYTHON_TYPE_NAMES: tuple[tuple[type[Any], str], ...] = (
    (bool, "boolean"),
    (int, "integer"),
    (float, "number"),
    (str, "string"),
    (list, "array"),
    (dict, "object"),
)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    for python_type, name in _PYTHON_TYPE_NAMES:
        if isinstance(value, python_type):
            return name
    return "unknown"


def _validate_into(instance: Any, schema: Any, path: str, errors: list[SchemaError]) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        errors.append(SchemaError(path, "no value is permitted by this schema"))
        return
    if not isinstance(schema, dict):
        errors.append(SchemaError(path, f"malformed schema fragment {schema!r}"))
        return

    expected = schema.get("type")
    if expected is not None:
        names = (expected,) if isinstance(expected, str) else tuple(expected)
        if not any(_matches_type(instance, name) for name in names):
            errors.append(
                SchemaError(path, f"expected {'|'.join(names)}, got {_type_name(instance)}")
            )
            return

    if "enum" in schema and instance not in schema["enum"]:
        errors.append(SchemaError(path, f"{instance!r} is not one of {schema['enum']!r}"))

    if isinstance(instance, dict):
        _validate_object(instance, schema, path, errors)
    elif isinstance(instance, list):
        _validate_array(instance, schema, path, errors)
    elif isinstance(instance, (int, float)) and not isinstance(instance, bool):
        _validate_number(instance, schema, path, errors)
    elif isinstance(instance, str):
        _validate_string(instance, schema, path, errors)


def _matches_type(instance: Any, name: str) -> bool:
    allowed = _JSON_TYPES.get(name)
    if allowed is None:
        return True
    if name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if name == "boolean":
        return isinstance(instance, bool)
    return isinstance(instance, allowed)


def _validate_object(
    instance: dict[str, Any], schema: dict[str, Any], path: str, errors: list[SchemaError]
) -> None:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(SchemaError(f"{path}.{name}", "required property is missing"))
        for name, value in instance.items():
            if name in properties:
                _validate_into(value, properties[name], f"{path}.{name}", errors)

    extra = schema.get("additionalProperties")
    if extra is False and isinstance(properties, dict):
        for name in instance:
            if name not in properties:
                errors.append(
                    SchemaError(f"{path}.{name}", "property is not allowed by this schema")
                )
    elif isinstance(extra, dict) and isinstance(properties, dict):
        for name, value in instance.items():
            if name not in properties:
                _validate_into(value, extra, f"{path}.{name}", errors)


def _validate_array(
    instance: list[Any], schema: dict[str, Any], path: str, errors: list[SchemaError]
) -> None:
    if "minItems" in schema and len(instance) < schema["minItems"]:
        errors.append(
            SchemaError(path, f"needs at least {schema['minItems']} items, got {len(instance)}")
        )
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        errors.append(
            SchemaError(path, f"allows at most {schema['maxItems']} items, got {len(instance)}")
        )
    items = schema.get("items")
    if isinstance(items, dict):
        for index, element in enumerate(instance):
            _validate_into(element, items, f"{path}[{index}]", errors)


def _validate_number(
    instance: float, schema: dict[str, Any], path: str, errors: list[SchemaError]
) -> None:
    if "minimum" in schema and instance < schema["minimum"]:
        errors.append(SchemaError(path, f"must be at least {schema['minimum']}"))
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        errors.append(SchemaError(path, f"must be greater than {schema['exclusiveMinimum']}"))
    if "maximum" in schema and instance > schema["maximum"]:
        errors.append(SchemaError(path, f"must be at most {schema['maximum']}"))
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        errors.append(SchemaError(path, f"must be less than {schema['exclusiveMaximum']}"))


def _validate_string(
    instance: str, schema: dict[str, Any], path: str, errors: list[SchemaError]
) -> None:
    if "minLength" in schema and len(instance) < schema["minLength"]:
        errors.append(SchemaError(path, f"must be at least {schema['minLength']} characters"))
    if "maxLength" in schema and len(instance) > schema["maxLength"]:
        errors.append(SchemaError(path, f"must be at most {schema['maxLength']} characters"))

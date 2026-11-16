"""Tool declaration types.

``ToolSpec`` is the wire-facing description of a capability: name, prose, and a
JSON Schema for its arguments. ``tools/registry.py`` is the single source of
truth for which specs are offered to a model; nothing else may construct the
tool list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model sees it.

    ``parameters`` is a JSON Schema object. It is sent verbatim, so a schema
    the endpoint cannot parse becomes a model-side failure rather than a
    local one -- keep them plain (objects, strings, integers, booleans,
    arrays of those).
    """

    name: str
    description: str
    parameters: dict[str, Any]

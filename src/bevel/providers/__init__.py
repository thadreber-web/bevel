"""Model providers.

Everything that talks to an inference server lives here, behind the
``ModelProvider`` protocol so the agent loop never learns about HTTP.

The seam is :mod:`bevel.providers.base`: import types from there. Only
:mod:`~bevel.providers.openai_compat` is allowed to know ``httpx`` exists, and
only :class:`~bevel.providers.registry.Registry` decides which profile to use.
"""

from bevel.providers.base import (
    APIError,
    AssistantTurn,
    Capabilities,
    ContentDelta,
    GenOptions,
    Health,
    MalformedResponseError,
    ModelProvider,
    ProviderError,
    ReasoningDelta,
    StreamAbortedError,
    StreamEnd,
    StreamEvent,
    TokenCount,
    TransportError,
    Usage,
    assemble,
)
from bevel.providers.mock import MockProvider, ScriptedTurn, tool_call
from bevel.providers.openai_compat import OpenAICompatProvider, build_request
from bevel.providers.registry import CachedProbe, Registry
from bevel.providers.repair import (
    RepairOutcome,
    ToolCallRepairer,
    extract_json_object,
    format_violations,
)

__all__ = [
    "APIError",
    "AssistantTurn",
    "CachedProbe",
    "Capabilities",
    "ContentDelta",
    "GenOptions",
    "Health",
    "MalformedResponseError",
    "MockProvider",
    "ModelProvider",
    "OpenAICompatProvider",
    "ProviderError",
    "ReasoningDelta",
    "Registry",
    "RepairOutcome",
    "ScriptedTurn",
    "StreamAbortedError",
    "StreamEnd",
    "StreamEvent",
    "TokenCount",
    "ToolCallRepairer",
    "TransportError",
    "Usage",
    "assemble",
    "build_request",
    "extract_json_object",
    "format_violations",
    "tool_call",
]

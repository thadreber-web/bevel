"""Turn what a server says it can do into tools the model is actually offered.

This is m10c. :class:`~bevel.mcp.stdio.StdioTransport` and
:class:`~bevel.mcp.http.HttpTransport` speak the wire and stop there -- they
deliberately import nothing from ``core/tools/``. This module is
where that boundary is allowed to be crossed: it takes a transport someone else
already started, asks it for ``tools/list``, and hands back objects satisfying
:class:`~bevel.core.tools.base.Tool`. Which transport it is, this module does not
ask; see :class:`StartedTransport`.

**The oracle is** ``tests/fixtures/mcp_server.py`` spawned for real by argv.
Every bridged tool in ``tests/unit/test_mcp_bridge.py`` is driven end to end
through a live child process; there is no stubbed transport in the suite. A
bridge tested against a fake transport proves the mapping and nothing about
whether the call actually reaches a server.

**Nothing here is checked for you.** ``tests/unit/test_tools_registry.py`` holds
the built-ins to a wire-safe name, ``type == "object"``,
``additionalProperties is False``, a description on every property, and no
keywords the validator ignores -- but it iterates ``registry.tools()``, and
bridged tools are deliberately not in the registry. So the bridge enforces the
same rules itself, by refusing any descriptor it cannot make satisfy them.

The decisions, and why:

* **Names are ``server__tool``, sanitised.** Lower-cased, every character
  outside ``[a-z0-9_]`` replaced with ``_``, leading and trailing underscores
  stripped, then checked against the ``^[a-z][a-z0-9_]*$`` the registry pins.
  This is not cosmetic: an unsanitised remote name is a schema the endpoint
  rejects at *offer* time, which takes every tool in the list down with it. Two
  worked examples -- ``Bad Name!`` becomes ``fixture__bad_name``, and ``my-tool``
  becomes ``fixture__my_tool``.
* **A collision refuses the second tool**, naming both originals. Silently
  keeping one would make which tool the model gets depend on the server's
  ordering; silently renaming the second would make the name on the wire
  unguessable.
* **A schema bevel cannot validate is refused, not stripped.** ``call_tool``
  validates against whatever is in ``parameters``, so a ``$ref`` or ``anyOf``
  that survives into the offered schema means the model may send anything and be
  told it was checked. Advertising a constraint that is not enforced is worse
  than not offering the tool.
* **``additionalProperties`` is forced to ``false``**, and ``properties`` is
  forced to a dict, because the validator only rejects extras when ``properties``
  is one. A remote schema that permits extras lets a hallucinated argument vanish
  silently.
* **A missing property description is filled in with an honest placeholder**
  rather than refused. The property's type is still enforceable, and losing a
  working tool over a documentation gap is the wrong trade; the placeholder says
  the server undocumented it rather than inventing a meaning bevel does not have.
* **Every bridged tool is ``EXEC_UNSAFE`` and serial, with no per-tool
  override.** The server is third-party code and the tier ladder says a tool
  cannot promote itself. ``EXEC_SAFE`` is out of reach by design because safety
  there comes from matching a specific argv against a rule, and a remote call has
  no argv. ``NETWORK_EGRESS`` would be the more honest label but has no producer
  today, so adopting it is its own decision with its own diff.
* **``isError`` is a result, never an exception.** A failed remote call is an
  observation the model reads and recovers from; aborting the iteration would
  punish the user for a server's bad day.
* **Teardown is not ours.** The bridge takes a transport someone else started and
  owns no lifecycle beyond issuing calls, so it has no ``close``.

Not built here, and the slice that delivers each: config and ``bevel mcp`` are
m10d, streamable-HTTP is m10e, and ``resources/list`` / ``resources/read`` are
m10f -- :mod:`bevel.mcp.resources`, which reuses :class:`StartedTransport` and
:func:`_render_blocks` from here rather than growing a second of either.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from bevel.core.tools.base import RiskTier, ToolContext, ToolError, ToolResult
from bevel.core.tools.schema import unsupported_keywords
from bevel.core.tools.types import ToolSpec
from bevel.mcp.stdio import McpHandshakeError

logger = logging.getLogger(__name__)

#: Characters an OpenAI-compatible endpoint accepts in a tool name. The extra
#: rule -- it must start with a letter -- is checked, not built.
_UNSAFE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9_]")

#: The shape ``tests/unit/test_tools_registry.py`` pins for every offered name.
_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

#: Fills a property description the server left out. Says what is true instead of
#: inventing a meaning bevel does not have.
_UNDOCUMENTED: Final[str] = (
    "Undocumented argument: the server declared this parameter but gave no description."
)


class StartedTransport(Protocol):
    """What the bridge needs from a transport that is already up.

    Both transports satisfy this and neither declares it: stdio has a child
    process and pipes, HTTP has a URL and a session id, and the bridge is
    supposed to care about neither. Naming the two members used -- one request
    and one error tail -- is the whole contract, so a third transport arrives
    without a change here and this module keeps importing neither of them.

    ``stderr_tail`` is stdio's name kept for both. It is the transport's honest
    answer to "what did the server last say that might explain this", which for
    HTTP is the body of its last error response; the bridge quotes it for either.
    """

    @property
    def stderr_tail(self) -> str: ...

    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | Sequence[Any] | None = None,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - a protocol signature, not a local deadline
    ) -> Any: ...


@dataclass(frozen=True)
class ToolRefusal:
    """One descriptor bevel refused to offer, and why.

    ``reason`` is self-contained -- it names the tool -- because m10d prints it on
    a line of its own. ``remote_name`` is ``None`` when the descriptor had no
    usable ``name`` member, the one failure that cannot be reported by name.
    """

    remote_name: str | None
    reason: str


class BridgedTool:
    """One remote tool, offered under a bevel name.

    ``remote_name`` is what goes on the wire and ``spec.name`` is what the model
    is told. Conflating them is the bug where every namespaced tool answers
    "unknown tool" and it looks like a fault in somebody else's server.
    """

    risk_tier: Final[RiskTier] = RiskTier.EXEC_UNSAFE
    is_parallel_safe: Final[bool] = False

    def __init__(
        self,
        transport: StartedTransport,
        server_name: str,
        remote_name: str,
        spec: ToolSpec,
    ) -> None:
        self._transport = transport
        self._server_name = server_name
        self.remote_name = remote_name
        self.spec = spec

    def __repr__(self) -> str:
        return f"BridgedTool({self.spec.name!r} -> {self._server_name!r}:{self.remote_name!r})"

    async def execute(self, arguments: Mapping[str, Any], ctx: ToolContext) -> ToolResult:
        """Ask the server to run it.

        ``ctx`` is deliberately unused: a remote call has no workspace to be
        jailed in, which is also why the tier is the highest one a tool may
        declare for itself.
        """
        del ctx
        params = {"name": self.remote_name, "arguments": dict(arguments)}
        try:
            result = await self._transport.call("tools/call", params)
        except Exception as exc:
            # Every transport failure -- a dead pipe, a timeout, a JSON-RPC error
            # object -- becomes an observation the model reads. CancelledError
            # derives from BaseException, so Ctrl+C still escapes.
            raise ToolError(
                self._say(f"the call did not complete: {type(exc).__name__}: {exc}")
            ) from exc

        if not isinstance(result, dict):
            raise ToolError(
                self._say(f"the server returned {type(result).__name__}, not a result object")
            )

        content = _render_blocks(result.get("content"))
        if result.get("isError") is True:
            return ToolResult.failure(
                self._say(content or "the server reported a failure without saying why")
            )
        return ToolResult(ok=True, content=content)

    def _say(self, reason: str) -> str:
        """Name the server, the remote tool, and whatever it wrote to stderr."""
        tail = self._transport.stderr_tail
        clause = (
            f"Last stderr from the server: {tail!r}"
            if tail
            else "The server wrote nothing to stderr."
        )
        return f"mcp server {self._server_name!r} tool {self.remote_name!r}: {reason}. {clause}"


@dataclass(frozen=True)
class BridgedServer:
    """What one server contributed: the tools it earned and the ones it did not.

    The refusals travel with the tools rather than being log-only, because m10d's
    ``bevel mcp list`` has to tell a user why a server they configured offers
    three tools instead of the seven it advertises.
    """

    server_name: str
    tools: tuple[BridgedTool, ...]
    refusals: tuple[ToolRefusal, ...]

    def by_name(self, name: str) -> BridgedTool:
        """Look up an offered tool by its namespaced name."""
        for tool in self.tools:
            if tool.spec.name == name:
                return tool
        raise KeyError(f"{self.server_name!r} does not offer a tool named {name!r}")


async def bridge_tools(transport: StartedTransport, server_name: str) -> BridgedServer:
    """Offer the tools of an already-started ``transport`` under ``server_name``.

    The caller owns the transport and must have started it; this issues its own
    ``tools/list`` and never spawns, restarts, or closes anything.

    A server that cannot answer ``tools/list`` at all raises the transport's own
    error -- that is a whole-server failure, and the caller (m10d) guards it with
    :class:`~bevel.mcp.stdio.McpTransportError`. A server whose *individual*
    descriptors are broken loses only those, and keeps its siblings.
    """
    result = await transport.call("tools/list", {})
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        raise McpHandshakeError(
            f"mcp server {server_name!r}: tools/list returned {result!r:.200}, which has no 'tools' array"
        )

    tools: list[BridgedTool] = []
    refusals: list[ToolRefusal] = []
    taken: dict[str, str] = {}

    for index, entry in enumerate(result["tools"]):
        tool, refusal = _bridge_one(transport, server_name, entry, taken, index)
        if refusal is not None:
            refusals.append(refusal)
            logger.warning("mcp server %r: refusing a tool -- %s", server_name, refusal.reason)
        elif tool is not None:
            taken[tool.spec.name] = tool.remote_name
            tools.append(tool)

    return BridgedServer(server_name=server_name, tools=tuple(tools), refusals=tuple(refusals))


def _bridge_one(
    transport: StartedTransport,
    server_name: str,
    entry: object,
    taken: Mapping[str, str],
    index: int,
) -> tuple[BridgedTool | None, ToolRefusal | None]:
    """Bridge one descriptor, or say why it cannot be offered.

    Exactly one of the two is ever set. Nothing here can raise into the caller's
    loop, which is the whole reason a broken descriptor costs one tool and not the
    server: every value below is untrusted, so each is checked before it reaches a
    constructor.

    ``index`` appears in the message only for a descriptor with no name to quote,
    so that "which one was it?" is still answerable from the log.
    """
    if not isinstance(entry, dict):
        return None, _refuse(
            None, f"tools/list entry {index} is {type(entry).__name__}, not an object"
        )

    raw_name = entry.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, _refuse(None, f"tools/list entry {index} has no usable 'name' member")

    description = entry.get("description")
    if not isinstance(description, str) or not description.strip():
        return None, _refuse(
            raw_name,
            "'description' is missing or blank, and the model cannot choose a tool it cannot read about",
        )

    parameters, problem = _parameters(entry.get("inputSchema"))
    if problem is not None:
        return None, _refuse(raw_name, problem)

    offered, why = _offered_name(raw_name, server_name, taken)
    if offered is None:
        return None, _refuse(raw_name, why or "could not be given a tool name")

    spec = ToolSpec(offered, description.strip(), parameters)
    return BridgedTool(transport, server_name, raw_name, spec), None


def _offered_name(
    raw_name: str, server_name: str, taken: Mapping[str, str]
) -> tuple[str | None, str | None]:
    """The namespaced name to offer, or why this descriptor does not get one.

    The collision branch is the reason ``taken`` maps to the *original* name: a
    refusal that only says "duplicate" cannot be acted on, because the server's
    own descriptors are what the user has to go and fix.
    """
    offered = f"{_sanitise(server_name)}__{_sanitise(raw_name)}"
    if not _NAME_RE.match(offered):
        return None, f"sanitises to {offered!r}, which is not a tool name an endpoint will accept"
    if offered in taken:
        return (
            None,
            f"sanitises to {offered!r}, the same offered name as {taken[offered]!r}; refusing the "
            "second rather than letting the server's ordering decide which one the model gets",
        )
    return offered, None


def _refuse(remote_name: str | None, reason: str) -> ToolRefusal:
    """A refusal whose message stands on its own when printed alone."""
    return ToolRefusal(remote_name, reason if remote_name is None else f"{remote_name}: {reason}")


def _parameters(schema: object) -> tuple[dict[str, Any], str | None]:
    """Build a ``parameters`` the validator fully enforces, or say why we cannot.

    The fallback for an absent ``inputSchema`` is a closed empty object, not
    ``{}`` -- the validator treats an empty schema as "anything passes", which is
    the opposite of what a missing schema should mean.
    """
    if schema is None:
        return {"type": "object", "properties": {}, "additionalProperties": False}, None

    if not isinstance(schema, dict):
        return {}, f"'inputSchema' is {type(schema).__name__}, not a JSON object"

    keywords = unsupported_keywords(schema)
    if keywords:
        listed = ", ".join(sorted(keywords))
        return (
            {},
            f"'inputSchema' uses keywords bevel does not enforce: {listed}. Offering it would "
            "advertise a constraint that is silently not applied, so the tool is refused instead",
        )

    declared = schema.get("properties")
    if not isinstance(declared, dict):
        declared = {}
    properties: dict[str, Any] = {}
    for name, declared_prop in declared.items():
        if not isinstance(declared_prop, dict):
            return (
                {},
                f"'inputSchema.properties.{name}' is {type(declared_prop).__name__}, not a schema",
            )
        # A property the server left undocumented still has an enforceable type,
        # and losing a working tool over a documentation gap is the wrong trade.
        described = (
            declared_prop
            if str(declared_prop.get("description") or "").strip()
            else {**declared_prop, "description": _UNDOCUMENTED}
        )
        properties[str(name)] = described

    required = [n for n in schema.get("required", []) if isinstance(n, str) and n in properties]
    parameters: dict[str, Any] = {
        # Forced, not copied: tools/call arguments are always a JSON object, so a
        # server declaring otherwise is describing a call it cannot receive.
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return parameters, None


def _render_blocks(blocks: object) -> str:
    """Content blocks to the single string ``ToolResult`` carries.

    A non-text block is named rather than dropped quietly, so the model knows the
    server sent something bevel could not show it.
    """
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        else:
            kind = block.get("type") if isinstance(block, dict) else None
            parts.append(f"[{kind or 'malformed'} content omitted by bevel]")
    return "\n".join(parts)


def _sanitise(name: str) -> str:
    """Reduce a remote name to the alphabet the endpoints accept.

    Deliberately does not collapse runs of ``_``: ``a b`` and ``a__b`` are two
    different tools on the server and must stay two different tools here.
    """
    return _UNSAFE.sub("_", name.lower()).strip("_")


__all__ = ["BridgedServer", "BridgedTool", "ToolRefusal", "bridge_tools"]

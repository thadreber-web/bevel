"""Typed configuration objects.

Kept separate from the loader so the shape of a valid config can be read (and
tested) without touching the filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class PolicyMode(StrEnum):
    """How much the engine is allowed to decide without asking.

    ``auto`` is the default because bevel is a coding agent that has to write
    files to be useful; it still asks for everything above a workspace write.
    ``ask`` makes even workspace writes require approval.
    """

    AUTO = "auto"
    ASK = "ask"


@dataclass(frozen=True)
class PolicyConfig:
    """The ``[policy]`` section: mode, extra rules, and where they are stored.

    ``allow`` and ``deny`` are command prefixes written the way you'd type them
    (``"git push"``, ``"rm -rf"``) and are matched against normalised argv.
    They extend the built-in tables; they cannot override them downwards, so an
    ``allow`` entry can never make a destructive command free.
    """

    mode: PolicyMode = PolicyMode.AUTO
    #: Path to ``policy.db``. ``None`` means ``~/.cache/bevel/policy.db``, the
    #: same cache directory the capability probe uses.
    db_path: Path | None = None
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryConfig:
    """The ``[memory]`` section: long-term recall, off unless switched on.

    ``vectors`` is opt-in and has no effect unless the endpoint actually
    answered a probe of ``/embeddings``. With it off, memory is lexical-only
    (FTS5 + bm25) and every recall says so rather than implying otherwise.
    """

    #: Off by default: storing memories about someone's codebase is a choice.
    enabled: bool = False
    #: Ask the endpoint for embeddings and re-rank candidates by cosine.
    vectors: bool = False
    #: Skip the approval that ``remember`` would otherwise raise on every save.
    #:
    #: ``remember`` is ``WRITE_OUTSIDE``, and that tier means always-ask -- it
    #: asks even in ``auto`` mode, which is correct for a write outside the
    #: workspace that is never deleted. But the model initiates those writes
    #: itself, and learned rules cannot cover it because they are keyed on argv
    #: and ``remember`` has none, so there is no way to teach the engine to
    #: expect them. This key is that way, and it is deliberately narrow: one
    #: tool name, off by default, and it does not reach a path that escapes the
    #: workspace jail.
    auto_accept: bool = False
    #: Memories handed back from one recall.
    limit: int = 8
    #: Hard ceiling on the injected block, counted in tokens, not characters.
    max_injection_tokens: int = 600


#: Seconds allowed for each phase of an MCP server's startup and each tool call.
DEFAULT_MCP_TIMEOUT = 30.0


@dataclass(frozen=True)
class McpServerConfig:
    """One ``[mcp.servers.<name>]`` table: how to reach one server.

    Exactly one of ``command`` or ``url`` is set, and the loader refuses a table
    that declares both or neither. ``command`` (with ``args``/``cwd``) is a stdio
    server bevel spawns; ``url`` is a streamable-HTTP server bevel speaks to over
    the network. The two are mutually exclusive because a process at a path and a
    process at a URL are different things to start, and silently preferring one
    would make a half-edited table mean whichever key the code happened to read
    first.

    ``command``/``args``/``cwd`` are used as argv directly -- never through a
    shell -- so there is no quoting to get wrong and nothing to inject. The
    server's name is the prefix on every tool it contributes (``fs__read``),
    which is why the loader insists it be a usable handle.

    ``env`` is *merged into* the parent environment rather than replacing it.
    A stdio server spawned with a bare ``env`` dict has no ``PATH``, which on
    Windows means most servers fail to start for a reason the user cannot see.
    Replacing the environment would be the more secure choice, but it turns a
    working config into a broken one, and bevel's sandboxing is not this
    layer's job. It means nothing to an ``url`` server, and the loader accepts it
    there rather than inventing a per-transport key check: it is simply never
    read.

    ``timeout`` is one number applied to all three phases of the transport
    (spawn, handshake, tool call). Splitting it into three keys would be three
    more things to configure and nobody has yet hit a case where they differ.
    """

    command: str | None = None
    #: A streamable-HTTP endpoint. Set instead of ``command``, never alongside it.
    url: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    #: A disabled server is still listed by ``bevel mcp list`` -- a server that
    #: silently vanished from the list is indistinguishable from a typo.
    enabled: bool = True
    timeout: float = DEFAULT_MCP_TIMEOUT


class ConfigError(Exception):
    """Configuration is present but unusable.

    Raised rather than defaulted: silently pointing at the wrong endpoint is
    worse than refusing to start, and the message names the file and key so the
    fix is obvious.
    """


@dataclass(frozen=True)
class ProfileConfig:
    """One endpoint.

    ``api_key_env`` names the variable holding the bearer token. The key itself
    is never stored in a config file -- a TOML in a repo is a secret waiting to
    be committed.
    """

    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    verify_tls: bool = True
    headers: dict[str, str] = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Read the key from the environment at call time.

        Resolved lazily so a config loaded before ``load_dotenv``-style setup
        still sees a key exported later.
        """
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        return self.api_key

    @property
    def tls_verified(self) -> bool:
        """False only for an explicit ``verify_tls = false`` in the profile."""
        return self.verify_tls


@dataclass(frozen=True)
class Config:
    """Everything bevel needs to reach a model, and how much to allow itself."""

    profiles: dict[str, ProfileConfig]
    default_profile: str
    #: Where the layers that produced this config were actually found, in
    #: increasing precedence. Reported by ``bevel doctor``.
    sources: tuple[Path, ...] = ()
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    #: MCP servers to spawn at the start of a run, keyed by the name used as
    #: their tool prefix.
    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)

    def profile(self, name: str | None = None) -> ProfileConfig:
        key = name or self.default_profile
        try:
            return self.profiles[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(f"no profile named {key!r} (available: {available})") from exc

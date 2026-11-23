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
    #: Memories handed back from one recall.
    limit: int = 8
    #: Hard ceiling on the injected block, counted in tokens, not characters.
    max_injection_tokens: int = 600


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

    def profile(self, name: str | None = None) -> ProfileConfig:
        key = name or self.default_profile
        try:
            return self.profiles[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(f"no profile named {key!r} (available: {available})") from exc

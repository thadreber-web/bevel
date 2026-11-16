"""Typed configuration objects.

Kept separate from the loader so the shape of a valid config can be read (and
tested) without touching the filesystem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0


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
    """Everything bevel needs to reach a model."""

    profiles: dict[str, ProfileConfig]
    default_profile: str
    #: Where the layers that produced this config were actually found, in
    #: increasing precedence. Reported by ``bevel doctor``.
    sources: tuple[Path, ...] = ()

    def profile(self, name: str | None = None) -> ProfileConfig:
        key = name or self.default_profile
        try:
            return self.profiles[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles)) or "none"
            raise ConfigError(f"no profile named {key!r} (available: {available})") from exc

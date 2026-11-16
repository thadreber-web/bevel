"""Endpoint profiles, health checks, and cached capability probes.

Probing costs requests, so it happens once per profile per process and the
result is remembered -- both in memory and, optionally, on disk. A profile that
was probed an hour ago is reused rather than re-sniffed, because the answer
describes a server configuration that changes only when the user changes it.

Staleness is explicit: :attr:`CachedProbe.probed_at` is kept so the TUI can say
"capabilities last verified 3 h ago" instead of implying the check just
happened.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from bevel.config.schema import ProfileConfig
from bevel.providers.base import Capabilities, Health
from bevel.providers.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 24 * 3600
"""How long a cached probe stays trustworthy. A day, because the thing being
described is a server deployment, not live state."""


@dataclass
class CachedProbe:
    """A probe result plus enough provenance to report how fresh it is."""

    profile: str
    probed_at: float
    capabilities: Capabilities
    models: tuple[str, ...] = ()

    def is_fresh(self, *, now: float | None = None, ttl: float = DEFAULT_CACHE_TTL_SECONDS) -> bool:
        age = (now if now is not None else time.time()) - self.probed_at
        return 0 <= age <= ttl

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.probed_at)


def _capabilities_from_dict(raw: dict[str, Any]) -> Capabilities:
    """Rebuild capabilities, ignoring keys from a newer bevel version.

    A cache written by a future version may carry fields this build does not
    know. Dropping them degrades gracefully; failing to start does not.
    """
    known = {f.name for f in fields(Capabilities)}
    return Capabilities(**{k: v for k, v in raw.items() if k in known})


class Registry:
    """Owns the providers for the configured profiles.

    Providers are created lazily and cached, so a process that only ever uses
    one profile never opens a connection pool for the others.
    """

    def __init__(
        self,
        profiles: dict[str, ProfileConfig],
        *,
        default_profile: str | None = None,
        cache_path: Path | None = None,
        clock: Any = time.time,
    ) -> None:
        if not profiles:
            raise ValueError("Registry requires at least one profile")
        self._profiles = dict(profiles)
        self._cache_path = cache_path
        self._clock = clock
        self._providers: dict[str, OpenAICompatProvider] = {}
        self._probes: dict[str, CachedProbe] = {}

        if default_profile is None:
            self.default_profile = next(iter(self._profiles))
        elif default_profile not in self._profiles:
            raise KeyError(f"unknown profile {default_profile!r}")
        else:
            self.default_profile = default_profile

        self._load_cache()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._profiles)

    def provider(self, name: str | None = None) -> OpenAICompatProvider:
        """The provider for ``name``, created on first use."""
        key = name or self.default_profile
        existing = self._providers.get(key)
        if existing is not None:
            return existing
        try:
            config = self._profiles[key]
        except KeyError as exc:
            raise KeyError(f"unknown profile {key!r}; have {sorted(self._profiles)}") from exc

        cached = self._probes.get(key)
        capabilities = (
            cached.capabilities if cached and cached.is_fresh(now=self._clock()) else None
        )
        provider = OpenAICompatProvider(
            name=config.name,
            base_url=config.base_url,
            model=config.model,
            api_key=config.resolve_api_key(),
            timeout=config.timeout,
            connect_timeout=config.connect_timeout,
            verify_tls=config.verify_tls,
            extra_headers=config.headers,
            capabilities=capabilities,
        )
        self._providers[key] = provider
        return provider

    async def aprobe(self, name: str | None = None, *, force: bool = False) -> CachedProbe:
        """Cached-or-fresh capabilities for ``name``."""
        key = name or self.default_profile
        cached = self._probes.get(key)
        if cached is not None and not force and cached.is_fresh(now=self._clock()):
            return cached

        provider = self.provider(key)
        capabilities = await provider.probe()
        health = await provider.health()
        probe = CachedProbe(
            profile=key,
            probed_at=self._clock(),
            capabilities=capabilities,
            models=health.models,
        )
        self._probes[key] = probe
        self._save_cache()
        return probe

    async def health(self, name: str | None = None) -> Health:
        """Liveness for one profile. Never generates tokens."""
        return await self.provider(name).health()

    async def health_all(self) -> dict[str, Health]:
        return {name: await self.health(name) for name in self._profiles}

    def cached_probe(self, name: str | None = None) -> CachedProbe | None:
        return self._probes.get(name or self.default_profile)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._providers.clear()

    # -- cache ------------------------------------------------------------

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.is_file():
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("ignoring unreadable capability cache %s: %s", self._cache_path, exc)
            return
        entries = raw.get("profiles") if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            return
        for name, entry in entries.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("capabilities"), dict):
                continue
            probed_at = entry.get("probed_at")
            if not isinstance(probed_at, (int, float)):
                continue
            models = entry.get("models")
            self._probes[name] = CachedProbe(
                profile=name,
                probed_at=float(probed_at),
                capabilities=_capabilities_from_dict(entry["capabilities"]),
                models=tuple(m for m in models if isinstance(m, str))
                if isinstance(models, list)
                else (),
            )

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        payload = {
            "version": 1,
            "profiles": {
                name: {
                    "probed_at": probe.probed_at,
                    "models": list(probe.models),
                    "capabilities": asdict(probe.capabilities),
                }
                for name, probe in self._probes.items()
            },
        }
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("could not write capability cache %s: %s", self._cache_path, exc)


__all__ = ["DEFAULT_CACHE_TTL_SECONDS", "CachedProbe", "Registry"]

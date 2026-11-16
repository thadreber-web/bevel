"""Tests for the profile registry and its capability cache.

The registry is the only thing that decides when the network gets touched for
capability discovery, so these tests pin down three promises: probing happens
once per profile while fresh, a cache written by another process is trusted,
and a corrupt cache degrades to a warning rather than a crash.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from bevel.config.schema import ProfileConfig
from bevel.providers import registry as registry_module
from bevel.providers.base import Capabilities, Health
from bevel.providers.registry import DEFAULT_CACHE_TTL_SECONDS, CachedProbe, Registry


def profile(name: str, **overrides: Any) -> ProfileConfig:
    defaults: dict[str, Any] = {
        "name": name,
        "base_url": f"https://{name}.example/v1",
        "model": f"{name}-model",
    }
    defaults.update(overrides)
    return ProfileConfig(**defaults)


CAPS = Capabilities(
    supports_tools=True,
    supports_parallel_tool_calls=True,
    max_model_len=131072,
    reasoning=True,
)


class FakeProvider:
    """Stands in for :class:`OpenAICompatProvider` so no socket is opened."""

    instances: list[FakeProvider] = []

    def __init__(self, *, name: str, base_url: str, model: str, **kwargs: Any) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.init_kwargs = kwargs
        self.capabilities: Capabilities = kwargs.get("capabilities") or Capabilities()
        self.probe_calls = 0
        self.health_calls = 0
        self.closed = 0
        self.probe_result = CAPS
        self.health_result = Health(ok=True, detail="ok", models=("a-model",), capabilities=CAPS)
        FakeProvider.instances.append(self)

    async def probe(self) -> Capabilities:
        self.probe_calls += 1
        return self.probe_result

    async def health(self) -> Health:
        self.health_calls += 1
        return self.health_result

    async def aclose(self) -> None:
        self.closed += 1


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Swap the provider class and reset the instance log between tests."""
    FakeProvider.instances = []
    monkeypatch.setattr(registry_module, "OpenAICompatProvider", FakeProvider)
    return FakeProvider


class Clock:
    """A clock the tests advance by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_registry(**kwargs: Any) -> Registry:
    kwargs.setdefault("profiles", {"default": profile("default")})
    return Registry(**kwargs)


class TestConstruction:
    def test_requires_at_least_one_profile(self) -> None:
        with pytest.raises(ValueError, match="at least one profile"):
            Registry({})

    def test_default_profile_is_the_first_one(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")})
        assert reg.default_profile == "a"

    def test_explicit_default_profile_is_honoured(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")}, default_profile="b")
        assert reg.default_profile == "b"

    def test_unknown_default_profile_is_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown profile 'nope'"):
            Registry({"a": profile("a")}, default_profile="nope")

    def test_names_lists_the_profiles(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")})
        assert reg.names == ("a", "b")


class TestProviderCreation:
    def test_provider_is_created_on_first_use(self) -> None:
        reg = make_registry()
        assert FakeProvider.instances == []
        reg.provider()
        assert len(FakeProvider.instances) == 1

    def test_provider_is_cached(self) -> None:
        reg = make_registry()
        assert reg.provider() is reg.provider()

    def test_each_profile_gets_its_own_provider(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")})
        assert reg.provider("a") is not reg.provider("b")

    def test_config_is_passed_through(self) -> None:
        reg = Registry(
            {
                "work": profile(
                    "work",
                    base_url="https://work.example/v1/",
                    model="wm",
                    timeout=30.0,
                    connect_timeout=2.0,
                    verify_tls=False,
                    headers={"X-Trace": "on"},
                )
            }
        )
        provider = reg.provider("work")
        assert provider.base_url == "https://work.example/v1"
        assert provider.model == "wm"
        assert provider.init_kwargs["timeout"] == 30.0
        assert provider.init_kwargs["connect_timeout"] == 2.0
        assert provider.init_kwargs["verify_tls"] is False
        assert provider.init_kwargs["extra_headers"] == {"X-Trace": "on"}

    def test_api_key_is_resolved_when_the_provider_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REG_KEY", "s3cret")
        reg = Registry({"p": profile("p", api_key_env="REG_KEY")})
        assert reg.provider("p").init_kwargs["api_key"] == "s3cret"

    def test_unknown_profile_names_the_available_ones(self) -> None:
        reg = Registry({"b": profile("b"), "a": profile("a")})
        with pytest.raises(KeyError, match=r"have \['a', 'b'\]"):
            reg.provider("missing")

    def test_a_fresh_cached_probe_seeds_the_provider(self) -> None:
        clock = Clock()
        reg = make_registry(cache_path=None, clock=clock)
        reg._probes["default"] = CachedProbe("default", clock.now, CAPS)
        assert reg.provider().capabilities is CAPS

    def test_a_stale_cached_probe_does_not(self) -> None:
        clock = Clock()
        reg = make_registry(cache_path=None, clock=clock)
        stale = clock.now - (DEFAULT_CACHE_TTL_SECONDS + 1)
        reg._probes["default"] = CachedProbe("default", stale, CAPS)
        assert reg.provider().capabilities == Capabilities()


class TestProbe:
    async def test_probe_returns_capabilities_and_models(self) -> None:
        reg = make_registry()
        probe = await reg.aprobe()
        assert probe.capabilities == CAPS
        assert probe.models == ("a-model",)

    async def test_probe_is_cached_in_memory(self) -> None:
        reg = make_registry()
        await reg.aprobe()
        await reg.aprobe()
        assert FakeProvider.instances[0].probe_calls == 1

    async def test_force_reprobes(self) -> None:
        reg = make_registry()
        await reg.aprobe()
        await reg.aprobe(force=True)
        assert FakeProvider.instances[0].probe_calls == 2

    async def test_a_stale_entry_is_reprobed(self) -> None:
        clock = Clock()
        reg = make_registry(clock=clock)
        await reg.aprobe()
        clock.advance(DEFAULT_CACHE_TTL_SECONDS + 1)
        await reg.aprobe()
        assert FakeProvider.instances[0].probe_calls == 2

    async def test_probe_records_the_time_from_the_injected_clock(self) -> None:
        clock = Clock(now=4242.0)
        reg = make_registry(clock=clock)
        probe = await reg.aprobe()
        assert probe.probed_at == 4242.0

    async def test_cached_probe_accessor_is_none_before_probing(self) -> None:
        reg = make_registry()
        assert reg.cached_probe() is None

    async def test_cached_probe_accessor_returns_the_probe(self) -> None:
        reg = make_registry()
        await reg.aprobe()
        assert reg.cached_probe() is not None

    async def test_cached_probe_for_another_profile(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")})
        await reg.aprobe("b")
        assert reg.cached_probe("b") is not None
        assert reg.cached_probe("a") is None


class TestCachedProbe:
    def test_fresh_within_ttl(self) -> None:
        assert CachedProbe("p", 100.0, CAPS).is_fresh(now=150.0, ttl=60.0) is True

    def test_stale_beyond_ttl(self) -> None:
        assert CachedProbe("p", 100.0, CAPS).is_fresh(now=200.0, ttl=60.0) is False

    def test_from_the_future_is_not_fresh(self) -> None:
        """A clock skew or a tampered file must not be trusted as "just probed"."""
        assert CachedProbe("p", 200.0, CAPS).is_fresh(now=100.0, ttl=60.0) is False

    def test_age_is_clamped_at_zero(self) -> None:
        """A probe timestamp in the future must not read as a negative age."""
        probe = CachedProbe("p", time.time() + 3600.0, CAPS)
        assert probe.age_seconds == 0.0

    def test_age_reports_the_real_age(self) -> None:
        probe = CachedProbe("p", time.time() - 120.0, CAPS)
        assert 119.0 <= probe.age_seconds <= 121.0

    def test_default_ttl_is_a_day(self) -> None:
        assert DEFAULT_CACHE_TTL_SECONDS == 86_400


class TestDiskCache:
    async def test_probe_is_written_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        reg = make_registry(cache_path=path)
        await reg.aprobe()
        assert path.is_file()

    async def test_round_trip_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        clock = Clock(now=500.0)
        first = Registry({"default": profile("default")}, cache_path=path, clock=clock)
        await first.aprobe()
        await first.aclose()

        FakeProvider.instances = []
        second = Registry({"default": profile("default")}, cache_path=path, clock=clock)
        probe = await second.aprobe()
        assert probe.probed_at == 500.0
        assert probe.capabilities == CAPS
        assert FakeProvider.instances == []

    async def test_a_stale_disk_entry_triggers_a_real_probe(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        clock = Clock(now=500.0)
        first = Registry({"default": profile("default")}, cache_path=path, clock=clock)
        await first.aprobe()
        await first.aclose()

        FakeProvider.instances = []
        clock.advance(DEFAULT_CACHE_TTL_SECONDS + 1)
        second = Registry({"default": profile("default")}, cache_path=path, clock=clock)
        assert second.cached_probe() is not None
        await second.aprobe()
        assert FakeProvider.instances[0].probe_calls == 1

    async def test_disk_cache_holds_every_profile(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        reg = Registry({"a": profile("a"), "b": profile("b")}, cache_path=path)
        await reg.aprobe("a")
        await reg.aprobe("b")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload["profiles"]) == {"a", "b"}

    async def test_saved_payload_is_versioned(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        reg = make_registry(cache_path=path)
        await reg.aprobe()
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1

    async def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "capabilities.json"
        reg = make_registry(cache_path=path)
        await reg.aprobe()
        assert path.is_file()

    def test_a_missing_cache_file_is_not_an_error(self, tmp_path: Path) -> None:
        reg = make_registry(cache_path=tmp_path / "absent.json")
        assert reg.cached_probe() is None

    def test_a_corrupt_cache_is_ignored_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text("{not json", encoding="utf-8")
        with caplog.at_level("WARNING", logger="bevel.providers.registry"):
            reg = make_registry(cache_path=path)
        assert reg.cached_probe() is None
        assert any("capability cache" in rec.message for rec in caplog.records)

    def test_a_cache_that_is_not_an_object_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert make_registry(cache_path=path).cached_probe() is None

    def test_a_cache_with_a_non_object_profiles_key_is_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text(json.dumps({"profiles": ["nope"]}), encoding="utf-8")
        assert make_registry(cache_path=path).cached_probe() is None

    @pytest.mark.parametrize(
        "entry",
        [
            {},
            {"capabilities": "nope"},
            {"capabilities": {"supports_tools": True}},
            {"probed_at": "yesterday", "capabilities": {}},
            {"probed_at": None, "capabilities": {}},
        ],
    )
    def test_malformed_entries_are_skipped(self, tmp_path: Path, entry: dict[str, Any]) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text(
            json.dumps({"profiles": {"default": entry}}),
            encoding="utf-8",
        )
        assert make_registry(cache_path=path).cached_probe() is None

    def test_unknown_capability_keys_from_a_newer_version_are_dropped(self, tmp_path: Path) -> None:
        """A cache from a future bevel must degrade, not prevent startup."""
        path = tmp_path / "capabilities.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "profiles": {
                        "default": {
                            "probed_at": 900.0,
                            "capabilities": {"supports_tools": True, "supports_mind_reading": True},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        probe = make_registry(cache_path=path).cached_probe()
        assert probe is not None
        assert probe.capabilities == Capabilities(supports_tools=True)

    def test_non_string_model_entries_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text(
            json.dumps(
                {
                    "profiles": {
                        "default": {
                            "probed_at": 900.0,
                            "capabilities": {},
                            "models": ["good", 3, None],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        probe = make_registry(cache_path=path).cached_probe()
        assert probe is not None
        assert probe.models == ("good",)

    def test_a_missing_models_key_yields_no_models(self, tmp_path: Path) -> None:
        path = tmp_path / "capabilities.json"
        path.write_text(
            json.dumps({"profiles": {"default": {"probed_at": 900.0, "capabilities": {}}}}),
            encoding="utf-8",
        )
        probe = make_registry(cache_path=path).cached_probe()
        assert probe is not None
        assert probe.models == ()

    def test_unwritable_cache_is_a_warning_not_a_failure(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        reg = make_registry(cache_path=blocker / "capabilities.json")
        with caplog.at_level("WARNING", logger="bevel.providers.registry"):
            reg._save_cache()
        assert any("could not write capability cache" in rec.message for rec in caplog.records)


class TestHealth:
    async def test_health_uses_the_default_profile(self) -> None:
        reg = make_registry()
        result = await reg.health()
        assert result.ok is True
        assert FakeProvider.instances[0].health_calls == 1

    async def test_health_all_covers_every_profile(self) -> None:
        reg = Registry({"a": profile("a"), "b": profile("b")})
        results = await reg.health_all()
        assert set(results) == {"a", "b"}
        assert all(result.ok for result in results.values())

    async def test_health_does_not_generate_tokens(self) -> None:
        """Health must not be the thing that spends the user's GPU time."""
        reg = make_registry()
        await reg.health()
        assert FakeProvider.instances[0].probe_calls == 0


class TestClose:
    async def test_close_closes_created_providers(self) -> None:
        reg = make_registry()
        provider = reg.provider()
        await reg.aclose()
        assert provider.closed == 1

    async def test_close_clears_the_cache_so_a_later_call_rebuilds(self) -> None:
        reg = make_registry()
        first = reg.provider()
        await reg.aclose()
        assert reg.provider() is not first

    async def test_close_without_any_provider_is_fine(self) -> None:
        await make_registry().aclose()

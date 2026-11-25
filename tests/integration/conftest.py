"""Fixtures for tests that talk to a real model server.

These tests are honest or they are worthless, so the rule here is: run only when a
real endpoint is configured *and* reachable, and say plainly which of the two
conditions failed when they do not. A suite that silently passes on a machine with
no GPU is indistinguishable from one that never ran.

Configuration comes entirely from the environment so that nothing about a particular
network -- a private address, a hostname -- ends up committed to a public
repository::

    $env:BEVEL_BASE_URL = "https://gpu.local:8000/v1"
    $env:BEVEL_MODEL    = "my-model"
    $env:VLLM_API_KEY   = "..."
    $env:BEVEL_INSECURE_TLS = "1"        # self-signed certificate
    uv run pytest tests/integration -v

The variable names are the ones :mod:`bevel.config.loader` already understands, so
"run the integration tests against X" and "run bevel against X" are the same
statement.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from bevel.providers.openai_compat import OpenAICompatProvider

BASE_URL_ENV = "BEVEL_BASE_URL"
MODEL_ENV = "BEVEL_MODEL"
API_KEY_ENVS = ("VLLM_API_KEY", "BEVEL_API_KEY")
INSECURE_TLS_ENV = "BEVEL_INSECURE_TLS"

TRUTHY = frozenset({"1", "true", "yes", "on"})
CONNECT_TIMEOUT_SECONDS = 5.0
# Generous, because the capability probe is a real generation on a real GPU and a
# reasoning model has to finish *thinking* before any of it arrives: probing this
# repo's endpoint costs ~700 reasoning tokens ahead of ~211 completion tokens, and
# at 60s that request intermittently times out. ``_ask`` fail-closes, so a timeout
# is reported as "this model cannot call tools" and every capability-gated test
# skips -- honestly, but on a coin flip, which makes the suite useless as a gate.
REQUEST_TIMEOUT_SECONDS = 180.0


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _api_key() -> str | None:
    for name in API_KEY_ENVS:
        if value := _env(name):
            return value
    return None


def _verify_tls() -> bool:
    return _env(INSECURE_TLS_ENV).lower() not in TRUTHY


def _missing_requirements() -> str | None:
    """Describe what has to be set before these tests can mean anything."""
    missing = [
        name
        for name, value in ((BASE_URL_ENV, _env(BASE_URL_ENV)), (MODEL_ENV, _env(MODEL_ENV)))
        if not value
    ]
    if _api_key() is None:
        missing.append(" or ".join(API_KEY_ENVS))
    if not missing:
        return None
    return "integration endpoint not configured (set " + ", ".join(missing) + ")"


def _host_and_port(base_url: str) -> tuple[str, int] | None:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def endpoint() -> AsyncIterator[OpenAICompatProvider]:
    """A probed provider for the configured live endpoint.

    Skips with a reason that distinguishes "not configured" from "configured but
    unreachable" -- the second is the one that would otherwise hide a regression.

    The probe is part of establishing the fixture, not something each test does
    for itself. ``capabilities`` starts pessimistic, so a provider that was never
    probed reports "no tools, no reasoning" and every capability-gated test skips
    while looking like it passed. Probing here means a skip means the endpoint
    genuinely lacks the feature. It is module-scoped because a probe is a handful
    of real generations and its answer does not change between tests.
    """
    if reason := _missing_requirements():
        pytest.skip(reason)

    base_url = _env(BASE_URL_ENV)
    model = _env(MODEL_ENV)

    # A TCP probe before anything else: an address that is simply absent from this
    # network fails in milliseconds instead of waiting on httpx.
    target = _host_and_port(base_url)
    if target is None:
        pytest.skip(f"{BASE_URL_ENV}={base_url!r} has no host to connect to")
    try:
        with socket.create_connection(target, timeout=CONNECT_TIMEOUT_SECONDS):
            pass
    except OSError as exc:
        pytest.skip(f"{target[0]}:{target[1]} is unreachable from this machine: {exc}")

    provider = OpenAICompatProvider(
        name="integration",
        base_url=base_url,
        model=model,
        api_key=_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        verify_tls=_verify_tls(),
    )
    try:
        health = await provider.health()
        if not health.ok:
            pytest.skip(f"{base_url.rstrip('/')}/models is not answering: {health.detail}")
        await provider.probe()
        yield provider
    finally:
        await provider.aclose()

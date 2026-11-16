"""Layered TOML configuration loading.

Precedence, lowest first::

    built-in defaults
      -> ~/.config/bevel/config.toml
      -> ./bevel.toml
      -> $BEVEL_CONFIG (an explicit file)
      -> environment variables
      -> keyword overrides (the flags layer, used by tests and the CLI)

Tables merge by profile name; scalars replace. A profile defined in two files is
merged key-by-key rather than replaced wholesale, so a project-level
``./bevel.toml`` can pin a model without restating the whole endpoint.
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from bevel.config.schema import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_TIMEOUT,
    Config,
    ConfigError,
    ProfileConfig,
)

logger = logging.getLogger(__name__)

CONFIG_DIRNAME = "bevel"
CONFIG_FILENAME = "config.toml"
PROJECT_FILENAME = "bevel.toml"
CAPABILITY_CACHE_FILENAME = "capabilities.json"

_PROFILE_KEYS = frozenset(
    {
        "base_url",
        "model",
        "api_key_env",
        "api_key",
        "timeout",
        "connect_timeout",
        "verify_tls",
        "headers",
    }
)


def config_home() -> Path:
    """``~/.config/bevel``.

    Used on every platform rather than the native app-data location: one
    documented path is easier to remember, easier to symlink into a dotfiles
    repo, and testable without mocking three environment variables.
    """
    return Path.home() / ".config" / CONFIG_DIRNAME


def cache_home() -> Path:
    """``~/.cache/bevel`` -- holds the capability probe cache."""
    return Path.home() / ".cache" / CONFIG_DIRNAME


def capability_cache_path() -> Path:
    return cache_home() / CAPABILITY_CACHE_FILENAME


def candidate_files(
    cwd: Path | None = None,
    *,
    explicit: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[Path]:
    """The config files that would be read, in increasing precedence.

    ``BEVEL_CONFIG`` is read from ``env`` when one is supplied, so a caller that
    injects its own environment (the tests, and ``load`` itself) sees the same
    answer it will get at runtime rather than whatever the process happens to
    have exported.
    """
    base = cwd if cwd is not None else Path.cwd()
    files = [config_home() / CONFIG_FILENAME, base / PROJECT_FILENAME]
    environ = os.environ if env is None else env
    override = explicit if explicit is not None else environ.get("BEVEL_CONFIG")
    if override:
        files.append(Path(override).expanduser())
    return files


def load(
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    explicit_file: str | os.PathLike[str] | None = None,
    profile: str | None = None,
) -> Config:
    """Load configuration. Raises :class:`ConfigError` if no usable profile exists.

    Refuses rather than inventing an endpoint: bevel talks only to servers the
    user configured, so a missing profile is a real error, not something to
    paper over with a localhost default that would hang for the connect timeout.
    """
    environ = dict(os.environ if env is None else env)
    profiles: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []

    for path in candidate_files(cwd, explicit=explicit_file, env=environ):
        if not path.is_file():
            continue
        try:
            data = _read_toml(path)
        except ConfigError:
            raise
        sources.append(path)
        _merge_profiles(profiles, data, path)

    _apply_env(profiles, environ)

    if not profiles:
        raise ConfigError(_no_config_message(cwd, explicit_file, environ))

    built = {name: _build_profile(name, raw) for name, raw in profiles.items()}
    default = _pick_default(profiles, environ, profile, sources)
    if default not in built:
        raise ConfigError(
            f"default profile {default!r} is not defined (defined: {', '.join(sorted(built))})"
        )
    return Config(profiles=built, default_profile=default, sources=tuple(sources))


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc


def _merge_profiles(target: dict[str, dict[str, Any]], data: dict[str, Any], path: Path) -> None:
    section = data.get("profile")
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [profile] must be a table of profiles")
    for name, raw in section.items():
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: profile {name!r} must be a table")
        unknown = set(raw) - _PROFILE_KEYS
        if unknown:
            raise ConfigError(
                f"{path}: profile {name!r} has unknown key(s) {', '.join(sorted(unknown))}; "
                f"expected one of {', '.join(sorted(_PROFILE_KEYS))}"
            )
        target.setdefault(name, {}).update(raw)


def _apply_env(profiles: dict[str, dict[str, Any]], env: dict[str, str]) -> None:
    """Environment overrides.

    ``BEVEL_BASE_URL``/``BEVEL_MODEL`` configure the profile named by
    ``BEVEL_PROFILE`` (defaulting to ``default``), which is how you point bevel
    at a one-off endpoint without writing a file.
    """
    name = env.get("BEVEL_PROFILE") or "default"
    overrides: dict[str, Any] = {}
    if base := env.get("BEVEL_BASE_URL"):
        overrides["base_url"] = base
    if model := env.get("BEVEL_MODEL"):
        overrides["model"] = model
    if key_env := env.get("BEVEL_API_KEY_ENV"):
        overrides["api_key_env"] = key_env
    if env.get("BEVEL_INSECURE_TLS") in {"1", "true", "yes", "on"}:
        logger.warning(
            "BEVEL_INSECURE_TLS is set: TLS certificate verification is disabled for profile %r",
            name,
        )
        overrides["verify_tls"] = False

    if overrides:
        profiles.setdefault(name, {}).update(overrides)


def _pick_default(
    profiles: dict[str, dict[str, Any]],
    env: dict[str, str],
    explicit: str | None,
    sources: list[Path],
) -> str:
    for candidate in (
        explicit,
        env.get("BEVEL_PROFILE"),
        _default_from_files(sources),
    ):
        if candidate:
            return candidate
    if len(profiles) == 1:
        return next(iter(profiles))
    return "default"


def _default_from_files(sources: list[Path]) -> str | None:
    for path in sources:
        try:
            data = _read_toml(path)
        except ConfigError:
            continue
        value = data.get("default_profile")
        if isinstance(value, str):
            return value
    return None


def _build_profile(name: str, raw: dict[str, Any]) -> ProfileConfig:
    base_url = raw.get("base_url")
    model = raw.get("model")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ConfigError(f"profile {name!r} is missing a string 'base_url'")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"profile {name!r} is missing a string 'model'")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(
            f"profile {name!r} has base_url {base_url!r}; it must start with http:// or https://"
        )

    headers = raw.get("headers", {})
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ConfigError(f"profile {name!r}: 'headers' must be a table of string to string")

    verify_tls = raw.get("verify_tls", True)
    if not isinstance(verify_tls, bool):
        raise ConfigError(f"profile {name!r}: 'verify_tls' must be a boolean")

    return ProfileConfig(
        name=name,
        base_url=base_url.strip().rstrip("/"),
        model=model.strip(),
        api_key_env=_optional_str(name, raw, "api_key_env"),
        api_key=_optional_str(name, raw, "api_key"),
        timeout=_positive_number(name, raw, "timeout"),
        connect_timeout=_positive_number(name, raw, "connect_timeout"),
        verify_tls=verify_tls,
        headers=dict(headers),
    )


def _optional_str(name: str, raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"profile {name!r}: {key!r} must be a string")
    return value


def _positive_number(name: str, raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if value is None:
        return DEFAULT_CONNECT_TIMEOUT if key == "connect_timeout" else DEFAULT_TIMEOUT
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"profile {name!r}: {key!r} must be a positive number")
    return float(value)


def _no_config_message(
    cwd: Path | None,
    explicit: str | os.PathLike[str] | None,
    env: dict[str, str] | None = None,
) -> str:
    searched = "\n".join(f"    {path}" for path in candidate_files(cwd, explicit=explicit, env=env))
    return (
        "no endpoint configured. Searched:\n"
        f"{searched}\n\n"
        "Create one with a [profile.<name>] table:\n\n"
        '  default_profile = "home"\n\n'
        "  [profile.home]\n"
        '  base_url = "https://gpu.local:8000/v1"\n'
        '  model = "my-model"\n'
        '  api_key_env = "MY_API_KEY"\n\n'
        "Or set BEVEL_BASE_URL and BEVEL_MODEL in the environment."
    )


__all__ = [
    "cache_home",
    "candidate_files",
    "capability_cache_path",
    "config_home",
    "load",
]

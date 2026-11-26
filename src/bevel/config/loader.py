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
import re
import tomllib
from pathlib import Path
from typing import Any

from bevel.config.schema import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_TIMEOUT,
    Config,
    ConfigError,
    McpServerConfig,
    MemoryConfig,
    PolicyConfig,
    PolicyMode,
    ProfileConfig,
)

logger = logging.getLogger(__name__)

CONFIG_DIRNAME = "bevel"
CONFIG_FILENAME = "config.toml"
PROJECT_FILENAME = "bevel.toml"
CAPABILITY_CACHE_FILENAME = "capabilities.json"
POLICY_DB_FILENAME = "policy.db"
SESSION_DB_FILENAME = "sessions.db"
MEMORY_DB_FILENAME = "memories.db"

_POLICY_KEYS = frozenset({"mode", "db_path", "allow", "deny"})
_POLICY_MODES = tuple(mode.value for mode in PolicyMode)

_MEMORY_KEYS = frozenset({"enabled", "vectors", "auto_accept", "limit", "max_injection_tokens"})

_MCP_KEYS = frozenset({"servers"})
_MCP_SERVER_KEYS = frozenset({"command", "url", "args", "env", "cwd", "enabled", "timeout"})

#: A server name becomes the tool prefix (``fs__read``) and the handle in
#: ``bevel mcp list``, so it has to survive both without sanitising.
_SERVER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

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


def policy_db_path() -> Path:
    """``~/.cache/bevel/policy.db`` -- learned allowlist and decision log."""
    return cache_home() / POLICY_DB_FILENAME


def data_home() -> Path:
    """``~/.local/share/bevel`` -- holds the session and memory databases.

    Same reasoning as :func:`config_home`: one documented path on every
    platform, correct on Windows too, where it is simply a directory under the
    user profile rather than the conventional ``%LOCALAPPDATA%`` one.
    """
    return Path.home() / ".local" / "share" / CONFIG_DIRNAME


def session_db_path() -> Path:
    """``~/.local/share/bevel/sessions.db`` -- transcripts and token accounting."""
    return data_home() / SESSION_DB_FILENAME


def memory_db_path() -> Path:
    """``~/.local/share/bevel/memories.db`` -- what bevel learned in past sessions.

    Deliberately outside any workspace: a memory is about the user and their
    projects, not about one checkout, and it survives a branch deletion. That
    is also why the ``remember`` tool is ``WRITE_OUTSIDE`` on the risk ladder.
    """
    return data_home() / MEMORY_DB_FILENAME


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

    policy_raw: dict[str, Any] = {}
    memory_raw: dict[str, Any] = {}
    mcp_raw: dict[str, dict[str, Any]] = {}
    for path in candidate_files(cwd, explicit=explicit_file, env=environ):
        if not path.is_file():
            continue
        try:
            data = _read_toml(path)
        except ConfigError:
            raise
        sources.append(path)
        _merge_profiles(profiles, data, path)
        _merge_policy(policy_raw, data, path)
        _merge_memory(memory_raw, data, path)
        _merge_mcp_servers(mcp_raw, data, path)

    _apply_env(profiles, environ)

    if not profiles:
        raise ConfigError(_no_config_message(cwd, explicit_file, environ))

    built = {name: _build_profile(name, raw) for name, raw in profiles.items()}
    default = _pick_default(profiles, environ, profile, sources)
    if default not in built:
        raise ConfigError(
            f"default profile {default!r} is not defined (defined: {', '.join(sorted(built))})"
        )
    return Config(
        profiles=built,
        default_profile=default,
        sources=tuple(sources),
        policy=_build_policy(policy_raw),
        memory=_build_memory(memory_raw),
        mcp_servers=_build_mcp_servers(mcp_raw),
    )


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


def _merge_policy(target: dict[str, Any], data: dict[str, Any], path: Path) -> None:
    """Fold one file's ``[policy]`` table into ``target``, validating as it goes.

    Validated per file rather than once at the end so the error names the file
    that actually contains the mistake. Later files replace earlier values key by
    key, which is how a project ``bevel.toml`` pins the mode without restating
    the user's allowlist.
    """
    section = data.get("policy")
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [policy] must be a table")
    unknown = set(section) - _POLICY_KEYS
    if unknown:
        raise ConfigError(
            f"{path}: [policy] has unknown key(s) {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(sorted(_POLICY_KEYS))}"
        )
    if "mode" in section:
        mode = section["mode"]
        if not isinstance(mode, str) or mode not in _POLICY_MODES:
            raise ConfigError(
                f"{path}: [policy] mode must be one of {', '.join(_POLICY_MODES)}, got {mode!r}"
            )
        target["mode"] = mode
    if "db_path" in section:
        raw = section["db_path"]
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigError(f"{path}: [policy] db_path must be a non-empty string")
        target["db_path"] = Path(raw).expanduser()
    for key in ("allow", "deny"):
        if key not in section:
            continue
        value = section[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ConfigError(f"{path}: [policy] {key} must be a list of non-empty strings")
        target[key] = tuple(value)


def _build_policy(raw: dict[str, Any]) -> PolicyConfig:
    return PolicyConfig(
        mode=PolicyMode(raw.get("mode", PolicyMode.AUTO)),
        db_path=raw.get("db_path"),
        allow=raw.get("allow", ()),
        deny=raw.get("deny", ()),
    )


def _merge_memory(target: dict[str, Any], data: dict[str, Any], path: Path) -> None:
    """Fold one file's ``[memory]`` table into ``target``, validating as it goes.

    Validated per file so the error names the file with the mistake, exactly as
    :func:`_merge_policy` does. ``enabled`` gates the whole feature, so a
    project ``bevel.toml`` turning it on is a decision the user made in that
    file and can point at.
    """
    section = data.get("memory")
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [memory] must be a table")
    unknown = set(section) - _MEMORY_KEYS
    if unknown:
        raise ConfigError(
            f"{path}: [memory] has unknown key(s) {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(sorted(_MEMORY_KEYS))}"
        )
    for key in ("enabled", "vectors", "auto_accept"):
        if key in section:
            if not isinstance(section[key], bool):
                raise ConfigError(f"{path}: [memory] {key} must be a boolean")
            target[key] = section[key]
    for key in ("limit", "max_injection_tokens"):
        if key not in section:
            continue
        value = section[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigError(f"{path}: [memory] {key} must be a positive integer")
        target[key] = value


def _build_memory(raw: dict[str, Any]) -> MemoryConfig:
    """Build from the merged table. ``_merge_memory`` validated every key already.

    Passing the dict through rather than restating each default means
    :class:`MemoryConfig` is the only place the defaults are written.
    """
    return MemoryConfig(**raw)


def _merge_mcp_servers(target: dict[str, dict[str, Any]], data: dict[str, Any], path: Path) -> None:
    """Fold one file's ``[mcp.servers]`` tables into ``target``, validating as it goes.

    A server is replaced wholesale rather than merged key by key, unlike a
    profile. The two tables mean different things: a profile is an endpoint
    described by independent settings, so pinning one key is meaningful on its
    own, but ``args`` only make sense alongside the ``command`` they belong to.
    Merging per key would let a project file keep a user's ``--api-key``
    argument while swapping the binary it is passed to.
    """
    section = data.get("mcp")
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(f"{path}: [mcp] must be a table")
    unknown = set(section) - _MCP_KEYS
    if unknown:
        raise ConfigError(
            f"{path}: [mcp] has unknown key(s) {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(sorted(_MCP_KEYS))}"
        )
    servers = section.get("servers")
    if servers is None:
        return
    if not isinstance(servers, dict):
        raise ConfigError(f"{path}: [mcp] servers must be a table")
    for name, raw in servers.items():
        if not isinstance(name, str) or not _SERVER_NAME_RE.fullmatch(name):
            raise ConfigError(
                f"{path}: mcp server name {name!r} is unusable; use letters, digits, '.', '_' or "
                "'-', starting with a letter or digit, since it becomes the tool prefix"
            )
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: mcp.servers.{name} must be a table")
        unknown = set(raw) - _MCP_SERVER_KEYS
        if unknown:
            raise ConfigError(
                f"{path}: mcp.servers.{name} has unknown key(s) {', '.join(sorted(unknown))}; "
                f"expected one of {', '.join(sorted(_MCP_SERVER_KEYS))}"
            )
        target[name] = _validated_server(path, name, raw)


def _validated_server(  # noqa: PLR0912 - one branch per configured key
    path: Path, name: str, raw: dict[str, Any]
) -> dict[str, Any]:
    """One server table, checked where it was written so the error can name the file.

    One branch per key rather than a loop over key/type pairs: the messages are
    all different, and a table of expectations would be a second language to
    learn to get six error strings.
    """
    out: dict[str, Any] = {}
    command = raw.get("command")
    url = raw.get("url")
    if command is not None and url is not None:
        raise ConfigError(
            f"{path}: mcp.servers.{name} declares both command and url; a server is either a "
            "process bevel spawns or an endpoint bevel speaks to, not both"
        )
    if command is None and url is None:
        raise ConfigError(f"{path}: mcp.servers.{name} must declare exactly one of command or url")
    if command is not None:
        if not isinstance(command, str) or not command.strip():
            raise ConfigError(f"{path}: mcp.servers.{name} command must be a non-empty string")
        out["command"] = command
    else:
        if not isinstance(url, str) or not url.strip():
            raise ConfigError(f"{path}: mcp.servers.{name} url must be a non-empty string")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"{path}: mcp.servers.{name} url must start with http:// or https://")
        out["url"] = url

    if "args" in raw:
        args = raw["args"]
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise ConfigError(f"{path}: mcp.servers.{name} args must be a list of strings")
        out["args"] = tuple(args)
    if "env" in raw:
        env = raw["env"]
        if not isinstance(env, dict):
            raise ConfigError(f"{path}: mcp.servers.{name} env must be a table")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ConfigError(f"{path}: mcp.servers.{name} env values must be strings")
        out["env"] = dict(env)
    if "cwd" in raw:
        cwd = raw["cwd"]
        if not isinstance(cwd, str) or not cwd.strip():
            raise ConfigError(f"{path}: mcp.servers.{name} cwd must be a string")
        out["cwd"] = Path(cwd).expanduser()
    if "enabled" in raw:
        enabled = raw["enabled"]
        if not isinstance(enabled, bool):
            raise ConfigError(f"{path}: mcp.servers.{name} enabled must be a boolean")
        out["enabled"] = enabled
    if "timeout" in raw:
        timeout = raw["timeout"]
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ConfigError(f"{path}: mcp.servers.{name} timeout must be a positive number")
        out["timeout"] = float(timeout)
    return out


def _build_mcp_servers(raw: dict[str, dict[str, Any]]) -> dict[str, McpServerConfig]:
    """Build from the merged tables; :func:`_merge_mcp_servers` validated them already."""
    return {name: McpServerConfig(**fields) for name, fields in raw.items()}


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
        "Create one with a [profile.<name>] table. Only base_url and model are\n"
        "required; base_url includes the /v1 suffix, and model must match what\n"
        "the server reports at <base_url>/models:\n\n"
        "  [profile.home]\n"
        '  base_url = "http://127.0.0.1:8000/v1"\n'
        '  model = "Qwen2.5-Coder-32B-Instruct"\n\n'
        "A local server usually needs no auth, so leave api_key_env out unless\n"
        "it actually requires a bearer token. See bevel.toml.example for every\n"
        "key, or skip the file entirely with BEVEL_BASE_URL and BEVEL_MODEL."
    )


__all__ = [
    "cache_home",
    "candidate_files",
    "capability_cache_path",
    "config_home",
    "data_home",
    "load",
    "memory_db_path",
    "policy_db_path",
    "session_db_path",
]

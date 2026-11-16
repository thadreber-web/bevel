"""Layered configuration: defaults -> ~/.config/bevel/config.toml -> ./bevel.toml -> env -> flags.

Later layers win. Secrets are never stored in these files -- a profile names an
environment variable (``api_key_env``) and the key is read from the environment
only when a provider is actually built, so ``bevel doctor`` can describe a
profile without the key ever being written down.
"""

from bevel.config.loader import (
    cache_home,
    candidate_files,
    capability_cache_path,
    config_home,
    load,
)
from bevel.config.schema import Config, ConfigError, ProfileConfig

__all__ = [
    "Config",
    "ConfigError",
    "ProfileConfig",
    "cache_home",
    "candidate_files",
    "capability_cache_path",
    "config_home",
    "load",
]

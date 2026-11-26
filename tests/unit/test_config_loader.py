"""Tests for the layered configuration loader.

The loader is the only place bevel decides where its model lives, so the tests
concentrate on precedence (which layer wins), on refusing rather than guessing,
and on error messages that name the file and key that need fixing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from bevel.config import loader
from bevel.config.loader import (
    CAPABILITY_CACHE_FILENAME,
    PROJECT_FILENAME,
    candidate_files,
    config_home,
    load,
)
from bevel.config.schema import (
    DEFAULT_CONNECT_TIMEOUT,
    DEFAULT_TIMEOUT,
    ConfigError,
    MemoryConfig,
)

MINIMAL = """
[profile.default]
base_url = "https://a.example/v1"
model = "model-a"
"""

#: Repository root, resolved from this file's known location.
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``Path.home()`` at a temp dir for every test in this module.

    Patching ``home`` rather than the loader's path functions means *every*
    derived path lands in the temp tree, so a test that forgets to qualify a
    helper call cannot write into the real ``~/.config/bevel`` on the machine
    running the suite. Autouse, because containment must not depend on each test
    remembering to ask for it.
    """
    base = tmp_path / "home"
    (base / ".config" / "bevel").mkdir(parents=True)
    (base / ".cache" / "bevel").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: base))


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def profile_toml(url: str = "https://a.example/v1", model: str = "model-a", **lines: Any) -> str:
    extra = "".join(f"{key} = {value}\n" for key, value in lines.items())
    return f"[profile.default]\nbase_url = {url!r}\nmodel = {model!r}\n{extra}"


class TestDefaults:
    def test_loads_a_single_profile(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        config = load(cwd=cwd, env={})
        assert config.default_profile == "default"
        assert config.profiles["default"].base_url == "https://a.example/v1"
        assert config.profiles["default"].model == "model-a"

    def test_sources_list_the_files_that_were_found(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        config = load(cwd=cwd, env={})
        assert config.sources == (cwd / PROJECT_FILENAME,)

    def test_missing_files_are_not_an_error(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        config = load(cwd=cwd, env={})
        assert loader.config_home() / "config.toml" not in config.sources

    def test_timeouts_default(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.timeout == DEFAULT_TIMEOUT
        assert profile.connect_timeout == DEFAULT_CONNECT_TIMEOUT

    def test_tls_is_verified_unless_told_otherwise(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        assert load(cwd=cwd, env={}).profiles["default"].tls_verified is True

    def test_trailing_slash_is_stripped_from_base_url(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml("https://a.example/v1/"))
        assert load(cwd=cwd, env={}).profiles["default"].base_url == "https://a.example/v1"


class TestPrecedence:
    def test_project_file_beats_user_file(self, cwd: Path) -> None:
        write(config_home() / "config.toml", profile_toml("https://user.example/v1", "user-model"))
        write(cwd / PROJECT_FILENAME, profile_toml("https://project.example/v1", "project-model"))
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.base_url == "https://project.example/v1"
        assert profile.model == "project-model"

    def test_profiles_merge_key_by_key_across_layers(self, cwd: Path) -> None:
        """A project file can pin a model without restating the whole endpoint."""
        write(
            loader.config_home() / "config.toml",
            profile_toml("https://user.example/v1", "user-model", api_key_env='"MY_KEY"'),
        )
        write(cwd / PROJECT_FILENAME, "[profile.default]\nmodel = 'project-model'\n")
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.model == "project-model"
        assert profile.base_url == "https://user.example/v1"
        assert profile.api_key_env == "MY_KEY"

    def test_explicit_file_beats_project_file(self, cwd: Path, tmp_path: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml("https://project.example/v1"))
        explicit = write(tmp_path / "other.toml", profile_toml("https://explicit.example/v1"))
        profile = load(cwd=cwd, env={}, explicit_file=explicit).profiles["default"]
        assert profile.base_url == "https://explicit.example/v1"

    def test_bevel_config_environment_variable_selects_a_file(
        self, cwd: Path, tmp_path: Path
    ) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml("https://project.example/v1"))
        explicit = write(tmp_path / "env.toml", profile_toml("https://from-env-file.example/v1"))
        profile = load(cwd=cwd, env={"BEVEL_CONFIG": str(explicit)}).profiles["default"]
        assert profile.base_url == "https://from-env-file.example/v1"

    def test_environment_beats_files(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(
            cwd=cwd,
            env={"BEVEL_BASE_URL": "https://env.example/v1", "BEVEL_MODEL": "env-model"},
        ).profiles["default"]
        assert profile.base_url == "https://env.example/v1"
        assert profile.model == "env-model"

    def test_candidate_files_are_in_increasing_precedence(self, cwd: Path, tmp_path: Path) -> None:
        explicit = tmp_path / "explicit.toml"
        files = candidate_files(cwd, explicit=explicit)
        assert files == [
            loader.config_home() / "config.toml",
            cwd / PROJECT_FILENAME,
            explicit,
        ]


class TestEnvironmentOverrides:
    def test_targets_bevel_profile_when_given(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        config = load(
            cwd=cwd,
            env={
                "BEVEL_PROFILE": "laptop",
                "BEVEL_BASE_URL": "https://laptop.example/v1",
                "BEVEL_MODEL": "laptop-model",
            },
        )
        assert config.profiles["laptop"].base_url == "https://laptop.example/v1"
        assert config.profiles["default"].base_url == "https://a.example/v1"

    def test_bevel_profile_selects_an_existing_profile(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            "[profile.default]\nbase_url = 'https://a.example/v1'\nmodel = 'a'\n"
            "[profile.laptop]\nbase_url = 'https://l.example/v1'\nmodel = 'l'\n",
        )
        assert load(cwd=cwd, env={"BEVEL_PROFILE": "laptop"}).default_profile == "laptop"

    def test_bevel_profile_naming_an_undefined_profile_is_an_error(self, cwd: Path) -> None:
        """Pointing at a profile that does not exist must not silently fall back."""
        write(cwd / PROJECT_FILENAME, MINIMAL)
        with pytest.raises(ConfigError, match="'laptop' is not defined"):
            load(cwd=cwd, env={"BEVEL_PROFILE": "laptop"})

    def test_api_key_env_override(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(cwd=cwd, env={"BEVEL_API_KEY_ENV": "OTHER_KEY"}).profiles["default"]
        assert profile.api_key_env == "OTHER_KEY"

    def test_insecure_tls_disables_verification(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(cwd=cwd, env={"BEVEL_INSECURE_TLS": "1"}).profiles["default"]
        assert profile.verify_tls is False
        assert profile.tls_verified is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_insecure_tls_accepts_common_truthy_strings(self, cwd: Path, value: str) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(cwd=cwd, env={"BEVEL_INSECURE_TLS": value}).profiles["default"]
        assert profile.verify_tls is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "maybe"])
    def test_insecure_tls_leaves_verification_on_otherwise(self, cwd: Path, value: str) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        profile = load(cwd=cwd, env={"BEVEL_INSECURE_TLS": value}).profiles["default"]
        assert profile.verify_tls is True

    def test_insecure_tls_is_loud(self, cwd: Path, caplog: pytest.LogCaptureFixture) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        with caplog.at_level("WARNING", logger="bevel.config.loader"):
            load(cwd=cwd, env={"BEVEL_INSECURE_TLS": "1"})
        assert any("verification is disabled" in rec.message for rec in caplog.records)


class TestDefaultProfileSelection:
    def test_default_profile_key_in_a_file_is_honoured(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            'default_profile = "work"\n\n'
            "[profile.work]\nbase_url = 'https://w.example/v1'\nmodel = 'w'\n"
            "[profile.home]\nbase_url = 'https://h.example/v1'\nmodel = 'h'\n",
        )
        assert load(cwd=cwd, env={}).default_profile == "work"

    def test_explicit_argument_beats_the_file(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            'default_profile = "work"\n\n'
            "[profile.work]\nbase_url = 'https://w.example/v1'\nmodel = 'w'\n"
            "[profile.home]\nbase_url = 'https://h.example/v1'\nmodel = 'h'\n",
        )
        assert load(cwd=cwd, env={}, profile="home").default_profile == "home"

    def test_a_single_profile_becomes_the_default(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            "[profile.only]\nbase_url = 'https://o.example/v1'\nmodel = 'o'\n",
        )
        assert load(cwd=cwd, env={}).default_profile == "only"

    def test_defaulting_to_missing_profile_is_an_error(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            'default_profile = "typo"\n\n'
            "[profile.work]\nbase_url = 'https://w.example/v1'\nmodel = 'w'\n",
        )
        with pytest.raises(ConfigError, match="typo"):
            load(cwd=cwd, env={})

    def test_error_lists_the_profiles_that_do_exist(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            'default_profile = "typo"\n\n'
            "[profile.work]\nbase_url = 'https://w.example/v1'\nmodel = 'w'\n"
            "[profile.home]\nbase_url = 'https://h.example/v1'\nmodel = 'h'\n",
        )
        with pytest.raises(ConfigError, match="home, work"):
            load(cwd=cwd, env={})

    def test_config_profile_raises_for_an_unknown_name(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        config = load(cwd=cwd, env={})
        with pytest.raises(ConfigError, match="no profile named 'nope'"):
            config.profile("nope")


class TestRefusalToGuess:
    def test_no_config_anywhere_is_an_error(self, cwd: Path) -> None:
        with pytest.raises(ConfigError, match="no endpoint configured"):
            load(cwd=cwd, env={})

    def test_error_shows_the_paths_it_searched(self, cwd: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            load(cwd=cwd, env={})
        text = str(excinfo.value)
        assert str(loader.config_home() / "config.toml") in text
        assert str(cwd / PROJECT_FILENAME) in text

    def test_error_shows_how_to_fix_it(self, cwd: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            load(cwd=cwd, env={})
        assert "[profile." in str(excinfo.value)
        assert "bevel.toml.example" in str(excinfo.value)

    def test_the_shipped_example_loads(self, cwd: Path) -> None:
        """The example is the documented way in, so it must survive the validator.

        A commented-out key is invisible to the loader, so this proves only the
        uncommented shape is accepted -- which is exactly what a user copies.
        """
        example = REPO_ROOT / "bevel.toml.example"
        assert example.is_file(), "bevel.toml.example must ship with the package"
        write(cwd / PROJECT_FILENAME, example.read_text(encoding="utf-8"))
        config = load(cwd=cwd, env={})
        profile = config.profiles[config.default_profile]
        assert profile.base_url.startswith(("http://", "https://"))
        assert profile.model
        # No auth by default: a local vLLM/llama.cpp server needs no header.
        assert profile.resolve_api_key() is None

    def test_a_file_with_no_profiles_is_still_no_config(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, "unrelated = 1\n")
        with pytest.raises(ConfigError, match="no endpoint configured"):
            load(cwd=cwd, env={})


class TestValidation:
    def test_missing_base_url_names_the_profile(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, "[profile.default]\nmodel = 'm'\n")
        with pytest.raises(ConfigError, match=r"default.*base_url"):
            load(cwd=cwd, env={})

    def test_missing_model_names_the_profile(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, "[profile.default]\nbase_url = 'https://a.example/v1'\n")
        with pytest.raises(ConfigError, match=r"default.*model"):
            load(cwd=cwd, env={})

    def test_blank_base_url_is_rejected(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml("   "))
        with pytest.raises(ConfigError, match="base_url"):
            load(cwd=cwd, env={})

    @pytest.mark.parametrize("url", ["a.example/v1", "ftp://a.example/v1", "//a.example"])
    def test_base_url_must_be_http(self, cwd: Path, url: str) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(url))
        with pytest.raises(ConfigError, match=r"http:// or https://"):
            load(cwd=cwd, env={})

    def test_unknown_keys_are_rejected_with_a_suggestion(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(base_urlx='"https://a.example/v1"'))
        with pytest.raises(ConfigError, match=r"unknown key.*base_urlx"):
            load(cwd=cwd, env={})

    def test_unknown_key_error_names_the_file(self, cwd: Path) -> None:
        path = write(cwd / PROJECT_FILENAME, profile_toml(nonsense="1"))
        with pytest.raises(ConfigError, match=re.escape(str(path))):
            load(cwd=cwd, env={})

    def test_profile_section_must_be_a_table(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, 'profile = "nope"\n')
        with pytest.raises(ConfigError, match=r"\[profile\] must be a table"):
            load(cwd=cwd, env={})

    def test_profile_must_be_a_table(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, 'profile.default = "nope"\n')
        with pytest.raises(ConfigError, match="must be a table"):
            load(cwd=cwd, env={})

    def test_invalid_toml_names_the_file(self, cwd: Path) -> None:
        path = write(cwd / PROJECT_FILENAME, "[profile.default\nbase_url = ")
        with pytest.raises(ConfigError, match="not valid TOML") as excinfo:
            load(cwd=cwd, env={})
        assert str(path) in str(excinfo.value)

    def test_headers_must_be_string_to_string(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(headers="{ X = 1 }"))
        with pytest.raises(ConfigError, match="headers"):
            load(cwd=cwd, env={})

    def test_headers_are_copied(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(headers='{ "X-Trace" = "on" }'))
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.headers == {"X-Trace": "on"}

    def test_verify_tls_must_be_boolean(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(verify_tls='"false"'))
        with pytest.raises(ConfigError, match=r"verify_tls.*boolean"):
            load(cwd=cwd, env={})

    def test_verify_tls_false_is_honoured(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(verify_tls="false"))
        assert load(cwd=cwd, env={}).profiles["default"].verify_tls is False

    @pytest.mark.parametrize("value", ["0", "-1", "true", '"10"'])
    def test_timeout_must_be_a_positive_number(self, cwd: Path, value: str) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(timeout=value))
        with pytest.raises(ConfigError, match=r"timeout.*positive number"):
            load(cwd=cwd, env={})

    def test_connect_timeout_is_separate_from_timeout(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            profile_toml(timeout="30", connect_timeout="2.5"),
        )
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert (profile.timeout, profile.connect_timeout) == (30.0, 2.5)

    def test_api_key_env_must_be_a_string(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(api_key_env="12"))
        with pytest.raises(ConfigError, match=r"api_key_env.*must be a string"):
            load(cwd=cwd, env={})


class TestApiKeyResolution:
    def test_key_comes_from_the_named_variable(
        self, cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(api_key_env='"MY_KEY"'))
        monkeypatch.setenv("MY_KEY", "s3cret")
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.resolve_api_key() == "s3cret"

    def test_inline_key_is_the_fallback(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(api_key='"inline"'))
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.resolve_api_key() == "inline"

    def test_environment_wins_over_the_inline_key(
        self, cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(api_key_env='"MY_KEY"', api_key='"inline"'))
        monkeypatch.setenv("MY_KEY", "from-env")
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.resolve_api_key() == "from-env"

    def test_blank_environment_value_falls_back_to_inline(
        self, cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write(cwd / PROJECT_FILENAME, profile_toml(api_key_env='"MY_KEY"', api_key='"inline"'))
        monkeypatch.setenv("MY_KEY", "")
        profile = load(cwd=cwd, env={}).profiles["default"]
        assert profile.resolve_api_key() == "inline"

    def test_no_key_configured_is_none(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        assert load(cwd=cwd, env={}).profiles["default"].resolve_api_key() is None

    def test_key_is_read_at_call_time_not_load_time(
        self, cwd: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key exported after ``load`` still resolves -- dotenv loads late."""
        write(cwd / PROJECT_FILENAME, profile_toml(api_key_env='"LATE_KEY"'))
        config = load(cwd=cwd, env={})
        assert config.profiles["default"].resolve_api_key() is None
        monkeypatch.setenv("LATE_KEY", "late")
        assert config.profiles["default"].resolve_api_key() == "late"


class TestPaths:
    def test_config_home_is_the_documented_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/home/me")))
        assert config_home() == Path("/home/me/.config/bevel")

    def test_capability_cache_path_is_under_the_cache_home(self) -> None:
        assert loader.capability_cache_path().name == CAPABILITY_CACHE_FILENAME

    def test_candidate_files_falls_back_to_the_real_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "cwd", classmethod(lambda _cls: Path("/nowhere")))
        files = candidate_files(None)
        assert files[1] == Path("/nowhere") / PROJECT_FILENAME

    def test_memory_db_path_is_under_the_data_home(self) -> None:
        assert loader.memory_db_path().name == loader.MEMORY_DB_FILENAME
        assert loader.memory_db_path().parent == loader.data_home()


class TestMemoryTable:
    """The ``[memory]`` table: off by default, validated per file, layered key by key."""

    def test_memory_is_off_when_no_table_is_present(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        memory = load(cwd=cwd, env={}).memory
        assert memory.enabled is False
        assert memory.vectors is False

    def test_a_table_turns_the_feature_on(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\nenabled = true\n")
        assert load(cwd=cwd, env={}).memory.enabled is True

    def test_defaults_come_from_the_schema_not_a_restated_copy(self, cwd: Path) -> None:
        """Only the keys the file sets are overridden; the rest stay at their defaults."""
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\nenabled = true\n")
        memory = load(cwd=cwd, env={}).memory
        assert memory.limit == MemoryConfig().limit
        assert memory.max_injection_tokens == MemoryConfig().max_injection_tokens

    def test_project_file_overrides_only_the_keys_it_sets(self, cwd: Path) -> None:
        write(
            loader.config_home() / "config.toml",
            f"{MINIMAL}\n[memory]\nenabled = true\nlimit = 12\n",
        )
        write(cwd / PROJECT_FILENAME, "[memory]\nlimit = 3\n")
        memory = load(cwd=cwd, env={}).memory
        assert memory.enabled is True
        assert memory.limit == 3

    def test_auto_accept_defaults_to_off(self, cwd: Path) -> None:
        """Skipping the ``WRITE_OUTSIDE`` approval has to be an explicit choice."""
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\nenabled = true\n")
        assert load(cwd=cwd, env={}).memory.auto_accept is False

    def test_auto_accept_can_be_turned_on(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            f"{MINIMAL}\n[memory]\nenabled = true\nauto_accept = true\n",
        )
        assert load(cwd=cwd, env={}).memory.auto_accept is True

    @pytest.mark.parametrize("key", ["enabled", "vectors", "auto_accept"])
    def test_boolean_keys_reject_non_booleans(self, cwd: Path, key: str) -> None:
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\n{key} = 'yes'\n")
        with pytest.raises(ConfigError, match=rf"\[memory\].*{key} must be a boolean"):
            load(cwd=cwd, env={})

    @pytest.mark.parametrize("key", ["limit", "max_injection_tokens"])
    @pytest.mark.parametrize("value", ["0", "-1", "'8'", "1.5", "true"])
    def test_integer_keys_reject_non_positive_integers(
        self, cwd: Path, key: str, value: str
    ) -> None:
        """``true`` is rejected too: TOML booleans are Python ints, and a flag is not a budget."""
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\n{key} = {value}\n")
        with pytest.raises(ConfigError, match=rf"\[memory\].*{key} must be a positive integer"):
            load(cwd=cwd, env={})

    def test_unknown_keys_are_rejected_with_the_expected_list(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\nenableda = true\n")
        with pytest.raises(ConfigError, match=r"unknown key\(s\).*enableda"):
            load(cwd=cwd, env={})

    def test_unknown_key_error_names_the_file(self, cwd: Path) -> None:
        path = write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[memory]\nnonsense = 1\n")
        with pytest.raises(ConfigError, match=re.escape(str(path))):
            load(cwd=cwd, env={})

    def test_memory_section_must_be_a_table(self, cwd: Path) -> None:
        """Written before the profile table, or TOML would nest it inside the profile."""
        write(cwd / PROJECT_FILENAME, f"memory = 'nope'\n{MINIMAL}")
        with pytest.raises(ConfigError, match=r"\[memory\] must be a table"):
            load(cwd=cwd, env={})


def mcp_table(name: str = "fixture", **fields: str) -> str:
    """A ``[mcp.servers.<name>]`` table body from already-quoted TOML values."""
    lines = [f"[mcp.servers.{name}]"]
    lines.extend(f"{key} = {value}" for key, value in fields.items())
    return "\n".join(lines)


def with_servers(*tables: str) -> str:
    """A whole profile file: the minimal profile plus the given server tables."""
    return "\n".join((MINIMAL, *tables))


#: A quoted TOML string, so a test can say ``toml_str("srv")`` rather than
#: carrying a backslash-escaped quote through an f-string (illegal on 3.11).
def toml_str(value: str) -> str:
    return f'"{value}"'


class TestMcpServersTable:
    """``[mcp.servers.<name>]``: the six keys, and nothing else."""

    def test_no_servers_table_means_no_servers(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, MINIMAL)
        assert load(cwd=cwd, env={}).mcp_servers == {}

    def test_a_server_is_read_with_every_key(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            with_servers(
                mcp_table(
                    "fixture",
                    command=toml_str("python"),
                    args="['-m', 'server', '--x']",
                    env="{TOKEN = 'secret'}",
                    cwd=toml_str("D:/work"),
                    enabled="false",
                    timeout="3.5",
                )
            ),
        )
        server = load(cwd=cwd, env={}).mcp_servers["fixture"]
        assert server.command == "python"
        assert server.args == ("-m", "server", "--x")
        assert server.env == {"TOKEN": "secret"}
        assert server.cwd == Path("D:/work")
        assert server.enabled is False
        assert server.timeout == 3.5

    def test_only_command_is_required(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, with_servers(mcp_table("a", command=toml_str("srv"))))
        server = load(cwd=cwd, env={}).mcp_servers["a"]
        assert server.command == "srv"
        assert server.args == ()
        assert server.env == {}
        assert server.cwd is None
        assert server.enabled is True
        assert server.timeout == 30.0

    def test_a_server_can_be_declared_by_url_instead_of_command(self, cwd: Path) -> None:
        """m10e: a remote server has no argv, so ``url`` stands in for ``command``."""
        write(
            cwd / PROJECT_FILENAME,
            with_servers(
                mcp_table(
                    "remote",
                    url=toml_str("https://mcp.test/rpc"),
                    env="{TOKEN = 'secret'}",
                    timeout="4.5",
                )
            ),
        )
        server = load(cwd=cwd, env={}).mcp_servers["remote"]
        assert server.url == "https://mcp.test/rpc"
        assert server.command is None
        assert server.args == ()
        assert server.env == {"TOKEN": "secret"}
        assert server.cwd is None
        assert server.enabled is True
        assert server.timeout == 4.5

    def test_command_and_url_are_mutually_exclusive(self, cwd: Path) -> None:
        """A table declaring both has no defined meaning; refusing it beats picking one."""
        write(
            cwd / PROJECT_FILENAME,
            with_servers(
                mcp_table("a", command=toml_str("srv"), url=toml_str("https://mcp.test/rpc"))
            ),
        )
        with pytest.raises(ConfigError, match="declares both command and url"):
            load(cwd=cwd, env={})

    def test_a_stdio_and_an_http_server_can_be_configured_together(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            with_servers(
                mcp_table("local", command=toml_str("one")),
                mcp_table("remote", url=toml_str("https://mcp.test/rpc")),
            ),
        )
        servers = load(cwd=cwd, env={}).mcp_servers
        assert servers["local"].command == "one"
        assert servers["remote"].url == "https://mcp.test/rpc"

    def test_several_servers_are_all_kept(self, cwd: Path) -> None:
        write(
            cwd / PROJECT_FILENAME,
            with_servers(
                mcp_table("a", command=toml_str("one")), mcp_table("b", command=toml_str("two"))
            ),
        )
        assert sorted(load(cwd=cwd, env={}).mcp_servers) == ["a", "b"]

    def test_a_later_file_replaces_the_same_server_wholesale(self, cwd: Path) -> None:
        """Per-key merging would leave a stale ``args`` behind when a project overrides a
        user's server. A table is one unit: the later file states all of it or none.
        """
        write(
            loader.config_home() / "config.toml",
            with_servers(
                mcp_table("a", command=toml_str("user-cmd"), args="['--keep-me']", env="{A = '1'}")
            ),
        )
        write(cwd / PROJECT_FILENAME, mcp_table("a", command=toml_str("project-cmd")))
        server = load(cwd=cwd, env={}).mcp_servers["a"]
        assert server.command == "project-cmd"
        assert server.args == ()
        assert server.env == {}

    def test_a_later_file_adds_a_different_server(self, cwd: Path) -> None:
        write(
            loader.config_home() / "config.toml",
            with_servers(mcp_table("user", command=toml_str("u"))),
        )
        write(cwd / PROJECT_FILENAME, mcp_table("project", command=toml_str("p")))
        assert sorted(load(cwd=cwd, env={}).mcp_servers) == ["project", "user"]

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (mcp_table("a", command="'c'", commandd="'d'"), "commandd"),
            (mcp_table("a", command="'c'", argss="[]"), "argss"),
            (mcp_table("a", command="'c'", envv="{}"), "envv"),
            (mcp_table("a", command="'c'", cwdd="'.'"), "cwdd"),
            (mcp_table("a", command="'c'", enable="true"), "enable"),
            (mcp_table("a", command="'c'", timeouts="1"), "timeouts"),
        ],
    )
    def test_unknown_keys_are_rejected_with_the_expected_list(
        self, cwd: Path, body: str, expected: str
    ) -> None:
        write(cwd / PROJECT_FILENAME, with_servers(body))
        with pytest.raises(ConfigError, match=rf"unknown key\(s\).*{expected}"):
            load(cwd=cwd, env={})

    def test_unknown_key_error_names_the_file_and_the_server(self, cwd: Path) -> None:
        path = write(
            cwd / PROJECT_FILENAME, with_servers(mcp_table("a", command="'c'", nonsense="1"))
        )
        with pytest.raises(ConfigError) as excinfo:
            load(cwd=cwd, env={})
        message = str(excinfo.value)
        assert str(path) in message
        assert "mcp.servers.a" in message
        assert "nonsense" in message

    @pytest.mark.parametrize("key", ["command", "args", "env", "cwd", "enabled", "timeout", "url"])
    def test_every_key_is_documented_in_the_example(self, key: str) -> None:
        """The example file is the only place a key is explained; an undocumented key is
        a key nobody finds."""
        example = (REPO_ROOT / "bevel.toml.example").read_text(encoding="utf-8")
        assert "[mcp.servers." in example
        assert re.search(rf"^#\s*{key} = ", example, re.MULTILINE)

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            (mcp_table("a", command="''"), r"command must be a non-empty string"),
            (mcp_table("a", command="'c'", command2="'d'"), "command2"),
            (mcp_table("a", command="'c'", args="'not a list'"), r"args must be a list of strings"),
            (mcp_table("a", command="'c'", args="[1]"), r"args must be a list of strings"),
            (mcp_table("a", command="'c'", env="'not a table'"), r"env must be a table"),
            (mcp_table("a", command="'c'", env="{N = 1}"), r"env values must be strings"),
            (mcp_table("a", command="'c'", cwd="1"), r"cwd must be a string"),
            (mcp_table("a", command="'c'", enabled="'yes'"), r"enabled must be a boolean"),
            (mcp_table("a", command="'c'", timeout="'30'"), r"timeout must be a positive number"),
            (mcp_table("a", command="'c'", timeout="0"), r"timeout must be a positive number"),
            (mcp_table("a", command="'c'", timeout="-1"), r"timeout must be a positive number"),
            (mcp_table("a", command="'c'", timeout="true"), r"timeout must be a positive number"),
            (mcp_table("a"), r"exactly one of command or url"),
            (mcp_table("a", url="''"), r"url must be a non-empty string"),
            (mcp_table("a", url="1"), r"url must be a non-empty string"),
            (mcp_table("a", url="'   '"), r"url must be a non-empty string"),
            (mcp_table("a", url="'mcp.test/rpc'"), r"url must start with http:// or https://"),
            (
                mcp_table("a", url="'ftp://mcp.test/rpc'"),
                r"url must start with http:// or https://",
            ),
        ],
    )
    def test_malformed_values_are_rejected(self, cwd: Path, body: str, expected: str) -> None:
        write(cwd / PROJECT_FILENAME, with_servers(body))
        with pytest.raises(ConfigError, match=expected):
            load(cwd=cwd, env={})

    @pytest.mark.parametrize("name", ["bad name", ""])
    def test_server_names_must_be_usable_handles(self, cwd: Path, name: str) -> None:
        """The name is the tool prefix and the handle in ``bevel mcp list``; a name with
        spaces or an empty name would produce an unusable tool name."""
        write(cwd / PROJECT_FILENAME, with_servers(f"[mcp.servers.{name!r}]\ncommand = 'c'"))
        with pytest.raises(ConfigError, match="server name"):
            load(cwd=cwd, env={})

    def test_the_mcp_section_must_be_a_table(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, f"mcp = 'nope'\n{MINIMAL}")
        with pytest.raises(ConfigError, match=r"\[mcp\] must be a table"):
            load(cwd=cwd, env={})

    def test_the_servers_key_must_be_a_table(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[mcp]\nservers = 'nope'\n")
        with pytest.raises(ConfigError, match=r"\[mcp\] servers must be a table"):
            load(cwd=cwd, env={})

    def test_a_server_table_must_be_a_table(self, cwd: Path) -> None:
        write(cwd / PROJECT_FILENAME, f"{MINIMAL}\n[mcp]\nservers.a = 'nope'\n")
        with pytest.raises(ConfigError, match=r"mcp\.servers\.a must be a table"):
            load(cwd=cwd, env={})

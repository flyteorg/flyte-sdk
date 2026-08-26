import os
import pathlib
import typing
from dataclasses import dataclass
from functools import lru_cache
from os import getenv
from pathlib import Path

import yaml

from flyte._logging import logger

# This is the default config file name for flyte
FLYTECTL_CONFIG_ENV_VAR = "FLYTECTL_CONFIG"
UCTL_CONFIG_ENV_VAR = "UCTL_CONFIG"
# Selects a named section under the config file's top-level `profiles:` key.
PROFILE_ENV_VAR = "FLYTE_PROFILE"

# The top-level YAML key holding named profiles.
PROFILES_KEY = "profiles"

# Process-wide profile selection, set by the CLI's `--profile` before any config is read.
# `None` means "consult FLYTE_PROFILE"; readers go through `get_active_profile()`.
_active_profile: typing.Optional[str] = None


def set_active_profile(profile: typing.Optional[str]) -> None:
    """
    Set the process-wide active profile.

    Called once by the CLI when `--profile` is passed, before any config is read. Library
    callers should prefer passing `profile=` explicitly to `flyte.config.auto()` instead of
    mutating process state.

    Args:
        profile: Profile name, or None to fall back to the `FLYTE_PROFILE` environment variable.
    """
    global _active_profile  # noqa: PLW0603
    _active_profile = profile


def get_active_profile() -> typing.Optional[str]:
    """
    The profile to apply when a caller does not pass one explicitly.

    Precedence: an explicit `set_active_profile()` (i.e. `--profile`) wins over the
    `FLYTE_PROFILE` environment variable. Returns None when neither is set, which reads the
    config file exactly as it did before profiles existed.
    """
    if _active_profile is not None:
        return _active_profile
    return os.environ.get(PROFILE_ENV_VAR) or None


class ProfileNotFoundError(ValueError):
    """Raised when a requested profile is not declared by the resolved config file."""


@dataclass
class YamlConfigEntry(object):
    """
    Creates a record for the config entry.
    Args:
        switch: dot-delimited string that should match flytectl args. Leaving it as dot-delimited instead of a list
          of strings because it's easier to maintain alignment with flytectl.
        config_value_type: Expected type of the value
    """

    switch: str
    config_value_type: typing.Type = str

    def get_env_name(self) -> str:
        var_name = self.switch.upper().replace(".", "_")
        return f"FLYTE_{var_name}"

    def read_from_env(self, transform: typing.Optional[typing.Callable] = None) -> typing.Optional[typing.Any]:
        """
        Reads the config entry from environment variable, the structure of the env var is current
        `FLYTE_{SECTION}_{OPTION}` all upper cased. We will change this in the future.
        """
        env = self.get_env_name()
        v = os.environ.get(env, None)
        if v is None:
            return None
        return transform(v) if transform else v

    def read_from_file(
        self, cfg: "ConfigFile", transform: typing.Optional[typing.Callable] = None
    ) -> typing.Optional[typing.Any]:
        if not cfg:
            return None
        try:
            v = cfg.get(self)
            if isinstance(v, bool) or bool(v is not None and v):
                return transform(v) if transform else v
        except Exception:
            ...

        return None


@dataclass
class ConfigEntry(object):
    """
    A top level Config entry holder, that holds multiple different representations of the config.
    Legacy means the INI style config files. YAML support is for the flytectl config file, which is there by default
    when flytectl starts a sandbox
    """

    yaml_entry: YamlConfigEntry
    transform: typing.Optional[typing.Callable[[str], typing.Any]] = None

    def read(self, cfg: typing.Optional["ConfigFile"] = None) -> typing.Optional[typing.Any]:
        """
        Reads the config Entry from the various sources in the following order,
        #. First try to read from the relevant environment variable,
        #. If missing, then try to read from the legacy config file, if one was parsed.
        #. If missing, then try to read from the yaml file.

        The constructor for ConfigFile currently does not allow specification of both the ini and yaml style formats.

        Args:
            cfg:
        """
        from_env = self.yaml_entry.read_from_env(self.transform)
        if from_env is not None:
            return from_env
        if cfg and cfg.yaml_config and self.yaml_entry:
            return self.yaml_entry.read_from_file(cfg, self.transform)

        return None


class ConfigFile(object):
    def __init__(self, location: str, profile: typing.Optional[str] = None):
        """
        Load the config from this location.

        Args:
            location: Path to the YAML config file.
            profile: Optional name of a section under the file's top-level `profiles:` key.
                When set, every lookup checks `profiles.<profile>.<switch>` first and falls back
                to the top-level `<switch>`, so the top level acts as shared defaults. When the
                file declares no `profiles:` key at all the profile is ignored entirely, which
                keeps pre-profile config files reading exactly as they did before.
        """
        self._location = location
        self._profile = profile
        self._yaml_config = self._read_yaml_config(location)

    @property
    def path(self) -> pathlib.Path:
        """
        Returns the path to the config file.

        Returns:
            Path to the config file
        """
        return pathlib.Path(self._location)

    @property
    def profile(self) -> typing.Optional[str]:
        """The profile this file is being read under, or None for the top level only."""
        return self._profile

    @property
    def profiles(self) -> typing.List[str]:
        """
        Names of the profiles declared under the file's top-level `profiles:` key.

        Returns an empty list when the file declares none, so callers can treat "no profiles"
        and "not a mapping" identically.
        """
        cfg = self._yaml_config
        if not isinstance(cfg, dict):
            return []
        profiles = cfg.get(PROFILES_KEY)
        if not isinstance(profiles, dict):
            return []
        return [str(k) for k in profiles]

    @staticmethod
    def _read_yaml_config(location: str | pathlib.Path) -> typing.Optional[typing.Dict[str, typing.Any]]:
        with open(location, "r") as fh:
            try:
                yaml_contents = yaml.safe_load(fh)
                return yaml_contents
            except yaml.YAMLError as exc:
                logger.warning(f"Error {exc} reading yaml config file at {location}, ignoring...")
                return None

    @staticmethod
    def _walk(root: typing.Any, keys: typing.Sequence[str]) -> typing.Any:
        """Follow a dot-delimited switch through nested mappings, or None if any hop is missing."""
        d = root
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def _get_from_yaml(self, c: YamlConfigEntry) -> typing.Any:
        keys = c.switch.split(".")  # flytectl switches are dot delimited
        root = self.yaml_config
        if self._profile:
            # Profile first, top level as the fallback: a profile overrides individual switches
            # without having to restate the whole file. A profile that resolves to None for this
            # switch (absent, or explicitly null) inherits the top-level value.
            scoped = self._walk(root, (PROFILES_KEY, self._profile, *keys))
            if scoped is not None:
                return scoped
        return self._walk(root, keys)

    def get(self, c: YamlConfigEntry) -> typing.Any:
        return self._get_from_yaml(c)

    @property
    def yaml_config(self) -> typing.Dict[str, typing.Any] | None:
        return self._yaml_config


def _config_path_from_git_root() -> pathlib.Path | None:
    from flyte.git import config_from_root

    config = config_from_root()
    if config is None:
        return None
    return config.source


def resolve_config_path() -> pathlib.Path | None:
    """
    Config is read from the following locations in order of precedence:
    1. ./config.yaml if it exists
    2. ./.flyte/config.yaml if it exists
    3. <git_root>/.flyte/config.yaml if it exists
    4. `UCTL_CONFIG` environment variable
    5. `FLYTECTL_CONFIG` environment variable
    6. ~/.union/config.yaml if it exists
    7. ~/.flyte/config.yaml if it exists
    """
    current_location_config = Path("config.yaml")
    if current_location_config.exists():
        return current_location_config
    logger.debug("No ./config.yaml found")

    dot_flyte_config = Path(".flyte", "config.yaml")
    if dot_flyte_config.exists():
        return dot_flyte_config
    logger.debug("No ./.flyte/config.yaml found")

    git_root_config = _config_path_from_git_root()
    if git_root_config:
        return git_root_config
    logger.debug("No .flyte/config.yaml found in git repo root")

    uctl_path_from_env = getenv(UCTL_CONFIG_ENV_VAR, None)
    if uctl_path_from_env:
        return pathlib.Path(uctl_path_from_env)
    logger.debug("No UCTL_CONFIG environment variable found, checking FLYTECTL_CONFIG")

    flytectl_path_from_env = getenv(FLYTECTL_CONFIG_ENV_VAR, None)
    if flytectl_path_from_env:
        return pathlib.Path(flytectl_path_from_env)
    logger.debug("No FLYTECTL_CONFIG environment variable found, checking default locations")

    home_dir_union_config = Path(Path.home(), ".union", "config.yaml")
    if home_dir_union_config.exists():
        return home_dir_union_config
    logger.debug("No ~/.union/config.yaml found, checking current directory")

    home_dir_flytectl_config = Path(Path.home(), ".flyte", "config.yaml")
    if home_dir_flytectl_config.exists():
        return home_dir_flytectl_config
    logger.debug("No ~/.flyte/config.yaml found, checking current directory")

    return None


@lru_cache
def _load_config_file(location: str | None, profile: str | None) -> ConfigFile | None:
    """
    Load and cache a `ConfigFile`. Split out from `get_config_file` so the cache key carries the
    profile: two profiles over the same path are different config files and must not share an
    entry. `location` of None means "search the default locations".
    """
    if location is None:
        config_path = resolve_config_path()
        if config_path is None:
            return None
        location = str(config_path)
    else:
        logger.debug(f"Using specified config file at {location}")

    cfg = ConfigFile(location, profile=profile)
    if profile and profile not in cfg.profiles:
        # Explicitly asked for a profile that isn't there. Failing loudly beats silently falling
        # back to the top-level defaults and talking to the wrong control plane.
        available = ", ".join(sorted(cfg.profiles)) or "none"
        raise ProfileNotFoundError(
            f"Profile {profile!r} not found in config file {location}. Available profiles: {available}."
        )
    return cfg


def get_config_file(
    c: typing.Union[str, pathlib.Path, ConfigFile, None],
    profile: typing.Optional[str] = None,
) -> ConfigFile | None:
    """
    Checks if the given argument is a file or a configFile and returns a loaded configFile else returns None

    Args:
        c: A path to a config file, an already-loaded `ConfigFile`, or None to search the default
            locations.
        profile: Optional profile to read the file under. When omitted the active profile
            (`--profile`, else `FLYTE_PROFILE`) applies. An already-loaded `ConfigFile` is
            returned as-is, since its profile was fixed when it was built.

    Raises:
        ProfileNotFoundError: if a profile was requested but the config file does not declare it.
    """
    if isinstance(c, ConfigFile):
        return c
    resolved = profile if profile is not None else get_active_profile()
    return _load_config_file(str(c) if c is not None else None, resolved)


def list_profiles(c: typing.Union[str, pathlib.Path, ConfigFile, None] = None) -> typing.List[str]:
    """
    Names of the profiles declared by a config file.

    Reads the file without applying a profile, so it works even when the active profile is
    invalid. Returns an empty list when no config file is found or none are declared.
    """
    if isinstance(c, ConfigFile):
        return c.profiles
    cfg = _load_config_file(str(c) if c is not None else None, None)
    return cfg.profiles if cfg else []


def read_file_if_exists(filename: typing.Optional[str], encoding=None) -> typing.Optional[str]:
    """
    Reads the contents of the file if passed a path. Otherwise, returns None.

    Args:
        filename: The file path to load
        encoding: The encoding to use when reading the file.

    Returns:
        The contents of the file as a string or None.
    """
    if not filename:
        return None

    file = pathlib.Path(filename)
    logger.debug(f"Reading file contents from \\[{file}] with current directory \\[{os.getcwd()}].")
    return file.read_text(encoding=encoding)

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
        ``FLYTE_{SECTION}_{OPTION}`` all upper cased. We will change this in the future.
        :return:
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

        :param cfg:
        :return:
        """
        from_env = self.yaml_entry.read_from_env(self.transform)
        if from_env is not None:
            return from_env
        if cfg and cfg.yaml_config and self.yaml_entry:
            return self.yaml_entry.read_from_file(cfg, self.transform)

        return None


class ConfigFile(object):
    def __init__(self, location: str):
        """
        Load the config from this location
        """
        self._location = location
        self._yaml_config = self._read_yaml_config(location)

    @property
    def path(self) -> pathlib.Path:
        """
        Returns the path to the config file.
        :return: Path to the config file
        """
        return pathlib.Path(self._location)

    @staticmethod
    def _read_yaml_config(location: str | pathlib.Path) -> typing.Optional[typing.Dict[str, typing.Any]]:
        with open(location, "r") as fh:
            try:
                yaml_contents = yaml.safe_load(fh)
                return yaml_contents
            except yaml.YAMLError as exc:
                logger.warning(f"Error {exc} reading yaml config file at {location}, ignoring...")
                return None

    def _get_from_yaml(self, c: YamlConfigEntry) -> typing.Any:
        keys = c.switch.split(".")  # flytectl switches are dot delimited
        d = typing.cast(typing.Dict[str, typing.Any], self.yaml_config)
        try:
            for k in keys:
                d = d[k]
            return d
        except KeyError:
            return None

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
def get_config_file(c: typing.Union[str, pathlib.Path, ConfigFile, None]) -> ConfigFile | None:
    """
    Checks if the given argument is a file or a configFile and returns a loaded configFile else returns None
    """
    if isinstance(c, (str, pathlib.Path)):
        logger.debug(f"Using specified config file at {c}")
        return ConfigFile(str(c))
    elif isinstance(c, ConfigFile):
        return c
    config_path = resolve_config_path()
    if config_path:
        return ConfigFile(str(config_path))
    return None


def read_file_if_exists(filename: typing.Optional[str], encoding=None) -> typing.Optional[str]:
    """
    Reads the contents of the file if passed a path. Otherwise, returns None.

    :param filename: The file path to load
    :param encoding: The encoding to use when reading the file.
    :return: The contents of the file as a string or None.
    """
    if not filename:
        return None

    file = pathlib.Path(filename)
    logger.debug(f"Reading file contents from [{file}] with current directory [{os.getcwd()}].")
    return file.read_text(encoding=encoding)

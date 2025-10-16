from __future__ import annotations

import dataclasses
import os
import pathlib
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import rich.repr

from flyte._logging import logger
from flyte.config import _internal
from flyte.config._reader import ConfigFile, get_config_file, read_file_if_exists

_all__ = ["ConfigFile", "PlatformConfig", "TaskConfig", "ImageConfig"]

if TYPE_CHECKING:
    from flyte.remote._client.auth import AuthType


@rich.repr.auto
@dataclass(init=True, repr=True, eq=True, frozen=True)
class PlatformConfig(object):
    """
    This object contains the settings to talk to a Flyte backend (the DNS location of your Admin server basically).

    :param endpoint: DNS for Flyte backend
    :param insecure: Whether or not to use SSL
    :param insecure_skip_verify: Whether to skip SSL certificate verification
    :param console_endpoint: endpoint for console if different from Flyte backend
    :param command: This command is executed to return a token using an external process
    :param proxy_command: This command is executed to return a token for proxy authorization using an external process
    :param client_id: This is the public identifier for the app which handles authorization for a Flyte deployment.
      More details here: https://www.oauth.com/oauth2-servers/client-registration/client-id-secret/.
    :param client_credentials_secret: Used for service auth, which is automatically called during pyflyte. This will
      allow the Flyte engine to read the password directly from the environment variable. Note that this is
      less secure! Please only use this if mounting the secret as a file is impossible
    :param scopes: List of scopes to request. This is only applicable to the client credentials flow
    :param auth_mode: The OAuth mode to use. Defaults to pkce flow
    :param ca_cert_file_path: [optional] str Root Cert to be loaded and used to verify admin
    :param http_proxy_url: [optional] HTTP Proxy to be used for OAuth requests
    """

    endpoint: str | None = None
    insecure: bool = False
    insecure_skip_verify: bool = False
    ca_cert_file_path: typing.Optional[str] = None
    console_endpoint: typing.Optional[str] = None
    command: typing.Optional[typing.List[str]] = None
    proxy_command: typing.Optional[typing.List[str]] = None
    client_id: typing.Optional[str] = None
    client_credentials_secret: typing.Optional[str] = None
    scopes: typing.List[str] = field(default_factory=list)
    auth_mode: "AuthType" = "Pkce"
    audience: typing.Optional[str] = None
    rpc_retries: int = 3
    http_proxy_url: typing.Optional[str] = None

    @classmethod
    def auto(cls, config_file: typing.Optional[typing.Union[str, ConfigFile]] = None) -> "PlatformConfig":
        """
        Reads from a config file, and overrides from Environment variables. Refer to ConfigEntry for details
        :param config_file:
        :return:
        """

        config_file = get_config_file(config_file)
        kwargs: typing.Dict[str, typing.Any] = {}
        kwargs = set_if_exists(kwargs, "insecure", _internal.Platform.INSECURE.read(config_file))
        kwargs = set_if_exists(
            kwargs, "insecure_skip_verify", _internal.Platform.INSECURE_SKIP_VERIFY.read(config_file)
        )
        kwargs = set_if_exists(kwargs, "ca_cert_file_path", _internal.Platform.CA_CERT_FILE_PATH.read(config_file))
        kwargs = set_if_exists(kwargs, "command", _internal.Credentials.COMMAND.read(config_file))
        kwargs = set_if_exists(kwargs, "proxy_command", _internal.Credentials.PROXY_COMMAND.read(config_file))
        kwargs = set_if_exists(kwargs, "client_id", _internal.Credentials.CLIENT_ID.read(config_file))

        is_client_secret = False
        client_credentials_secret = read_file_if_exists(
            _internal.Credentials.CLIENT_CREDENTIALS_SECRET_LOCATION.read(config_file)
        )
        if client_credentials_secret:
            is_client_secret = True
            if client_credentials_secret.endswith("\n"):
                logger.info("Newline stripped from client secret")
                client_credentials_secret = client_credentials_secret.strip()
        kwargs = set_if_exists(
            kwargs,
            "client_credentials_secret",
            client_credentials_secret,
        )

        client_credentials_secret_env_var = _internal.Credentials.CLIENT_CREDENTIALS_SECRET_ENV_VAR.read(config_file)
        if client_credentials_secret_env_var:
            client_credentials_secret = os.getenv(client_credentials_secret_env_var)
            if client_credentials_secret:
                is_client_secret = True
        kwargs = set_if_exists(kwargs, "client_credentials_secret", client_credentials_secret)
        kwargs = set_if_exists(kwargs, "scopes", _internal.Credentials.SCOPES.read(config_file))
        kwargs = set_if_exists(kwargs, "auth_mode", _internal.Credentials.AUTH_MODE.read(config_file))
        if is_client_secret:
            kwargs = set_if_exists(kwargs, "auth_mode", "ClientSecret")
        kwargs = set_if_exists(kwargs, "endpoint", _internal.Platform.URL.read(config_file))
        kwargs = set_if_exists(kwargs, "console_endpoint", _internal.Platform.CONSOLE_ENDPOINT.read(config_file))

        kwargs = set_if_exists(kwargs, "http_proxy_url", _internal.Platform.HTTP_PROXY_URL.read(config_file))
        return PlatformConfig(**kwargs)

    def replace(self, **kwargs: typing.Any) -> "PlatformConfig":
        """
        Returns a new PlatformConfig instance with the values from the kwargs overriding the current instance.
        """
        return dataclasses.replace(self, **kwargs)

    @classmethod
    def for_endpoint(cls, endpoint: str, insecure: bool = False) -> "PlatformConfig":
        return PlatformConfig(endpoint=endpoint, insecure=insecure)


@rich.repr.auto
@dataclass(init=True, repr=True, eq=True, frozen=True)
class TaskConfig(object):
    org: str | None = None
    project: str | None = None
    domain: str | None = None

    @classmethod
    def auto(cls, config_file: typing.Optional[typing.Union[str, ConfigFile]] = None) -> "TaskConfig":
        """
        Reads from a config file, and overrides from Environment variables. Refer to ConfigEntry for details
        :param config_file:
        :return:
        """
        config_file = get_config_file(config_file)
        kwargs: typing.Dict[str, typing.Any] = {}
        kwargs = set_if_exists(kwargs, "org", _internal.Task.ORG.read(config_file))
        kwargs = set_if_exists(kwargs, "project", _internal.Task.PROJECT.read(config_file))
        kwargs = set_if_exists(kwargs, "domain", _internal.Task.DOMAIN.read(config_file))
        return TaskConfig(**kwargs)


@rich.repr.auto
@dataclass(init=True, repr=True, eq=True, frozen=True)
class ImageConfig(object):
    """
    Configuration for Docker image settings.
    """

    builder: str | None = None
    image_refs: typing.Dict[str, str] = field(default_factory=dict)

    @classmethod
    def auto(cls, config_file: typing.Optional[typing.Union[str, ConfigFile]] = None) -> "ImageConfig":
        """
        Reads from a config file, and overrides from Environment variables. Refer to ConfigEntry for details
        :param config_file:
        :return:
        """
        config_file = get_config_file(config_file)
        kwargs: typing.Dict[str, typing.Any] = {}
        kwargs = set_if_exists(kwargs, "builder", _internal.Image.BUILDER.read(config_file))
        kwargs = set_if_exists(kwargs, "image_refs", _internal.Image.IMAGE_REFS.read(config_file))
        return ImageConfig(**kwargs)


@rich.repr.auto
@dataclass(init=True, repr=True, eq=True, frozen=True)
class Config(object):
    """
    This the parent configuration object and holds all the underlying configuration object types. An instance of
    this object holds all the config necessary to

    1. Interactive session with Flyte backend
    2. Some parts are required for Serialization, for example Platform Config is not required
    3. Runtime of a task
    """

    platform: PlatformConfig = field(default=PlatformConfig())
    task: TaskConfig = field(default=TaskConfig())
    image: ImageConfig = field(default=ImageConfig())
    source: pathlib.Path | None = None

    def with_params(
        self,
        platform: PlatformConfig | None = None,
        task: TaskConfig | None = None,
        image: ImageConfig | None = None,
    ) -> "Config":
        return Config(
            platform=platform or self.platform,
            task=task or self.task,
            image=image or self.image,
        )

    @classmethod
    def auto(cls, config_file: typing.Union[str, pathlib.Path, ConfigFile, None] = None) -> "Config":
        """
        Automatically constructs the Config Object. The order of precedence is as follows
          1. first try to find any env vars that match the config vars specified in the FLYTE_CONFIG format.
          2. If not found in environment then values ar read from the config file
          3. If not found in the file, then the default values are used.

        :param config_file: file path to read the config from, if not specified default locations are searched
        :return: Config
        """
        config_file = get_config_file(config_file)
        if config_file is None:
            logger.debug("No config file found, using default values")
            return Config()
        return Config(
            platform=PlatformConfig.auto(config_file),
            task=TaskConfig.auto(config_file),
            image=ImageConfig.auto(config_file),
            source=config_file.path,
        )


def set_if_exists(d: dict, k: str, val: typing.Any) -> dict:
    """
    Given a dict ``d`` sets the key ``k`` with value of config ``v``, if the config value ``v`` is set
    and return the updated dictionary.
    """
    exists = isinstance(val, bool) or bool(val is not None and val)
    if exists:
        d[k] = val
    return d


def auto(config_file: typing.Union[str, pathlib.Path, ConfigFile, None] = None) -> Config:
    """
    Automatically constructs the Config Object. The order of precedence is as follows
      1. If specified, read the config from the provided file path.
      2. If not specified, the config file is searched in the default locations.
            a. ./config.yaml if it exists  (current working directory)
            b. ./.flyte/config.yaml if it exists (current working directory)
            c. <git_root>/.flyte/config.yaml if it exists
            d. `UCTL_CONFIG` environment variable
            e. `FLYTECTL_CONFIG` environment variable
            f. ~/.union/config.yaml if it exists
            g. ~/.flyte/config.yaml if it exists
    3. If any value is not found in the config file, the default value is used.
    4. For any value there are environment variables that match the config variable names, those will override

    :param config_file: file path to read the config from, if not specified default locations are searched
    :return: Config
    """
    return Config.auto(config_file)

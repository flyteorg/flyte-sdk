import inspect
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Union

import rich.repr

from flyte import Environment
from flyte.app._input import Input
from flyte.app._types import Domain, Link, Port, Scaling
from flyte.models import SerializationContext

APP_NAME_RE = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")
INVALID_APP_PORTS = [8012, 8022, 8112, 9090, 9091]
INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR = "INTERNAL_APP_ENDPOINT_PATTERN"


@rich.repr.auto
@dataclass(init=True, repr=True)
class AppEnvironment(Environment):
    """
    :param name: Name of the environment
    :param image: Docker image to use for the environment. If set to "auto", will use the default image.
    :param resources: Resources to allocate for the environment.
    :param env_vars: Environment variables to set for the environment.
    :param secrets: Secrets to inject into the environment.
    :param depends_on: Environment dependencies to hint, so when you deploy the environment, the dependencies are
        also deployed. This is useful when you have a set of environments that depend on each other.
    """

    type: Optional[str] = None
    port: int | Port = 8080
    args: Optional[Union[List[str], str]] = None
    command: Optional[Union[List[str], str]] = None
    requires_auth: bool = True
    scaling: Scaling = field(default_factory=Scaling)
    domain: Domain | None = field(default_factory=Domain)
    # Integration
    links: List[Link] = field(default_factory=list)

    # Code
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)

    # queue / cluster_pool
    cluster_pool: str = "default"

    # config: Optional[AppConfigProtocol] = None

    def _validate_name(self):
        if not APP_NAME_RE.fullmatch(self.name):
            raise ValueError(
                f"App name '{self.name}' must consist of lower case alphanumeric characters or '-', "
                "and must start and end with an alphanumeric character."
            )

    def __post_init__(self):
        super().__post_init__()
        if self.args is not None and not isinstance(self.args, (list, str)):
            raise TypeError(f"Expected args to be of type List[str] or str, got {type(self.args)}")
        if isinstance(self.port, int):
            self.port = Port(port=self.port)  # Name should be blank can be h2c / http1
            if self.port.port in INVALID_APP_PORTS:
                raise ValueError(f"Port {self.port.port} is reserved and cannot be used for AppEnvironment")
        if self.command is not None and not isinstance(self.command, (list, str)):
            raise TypeError(f"Expected command to be of type List[str] or str, got {type(self.command)}")
        if not isinstance(self.scaling, Scaling):
            raise TypeError(f"Expected scaling to be of type Scaling, got {type(self.scaling)}")
        if not isinstance(self.domain, (Domain, type(None))):
            raise TypeError(f"Expected domain to be of type Domain or None, got {type(self.domain)}")
        for link in self.links:
            if not isinstance(link, Link):
                raise TypeError(f"Expected links to be of type List[Link], got {type(link)}")

        self._validate_name()

        # get instantiated file to keep track of app root directory
        frame = inspect.currentframe().f_back.f_back  # two frames up to get the app filename
        self._app_filename = frame.f_code.co_filename

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        if self.args is None:
            return []
        elif isinstance(self.args, str):
            return shlex.split(self.args)
        else:
            # args is a list
            return self.args

    def _serialize_inputs(self) -> str:
        if not self.inputs:
            return ""
        from ._input import SerializableInputCollection

        serialized_inputs = SerializableInputCollection.from_inputs(self.inputs)
        return serialized_inputs.to_transport

    def container_cmd(self, serialize_context: SerializationContext) -> List[str]:
        if self.command is None:
            # Default command
            version = serialize_context.version
            if version is None and serialize_context.code_bundle is not None:
                version = serialize_context.code_bundle.computed_version

            cmd: list[str] = [
                "fserve",
                "--version",
                version or "",
                "--project",
                serialize_context.project or "",
                "--domain",
                serialize_context.domain or "",
                "--org",
                serialize_context.org or "",
            ]

            if serialize_context.image_cache and serialize_context.image_cache.serialized_form:
                cmd = [*cmd, "--image-cache", serialize_context.image_cache.serialized_form]
            else:
                if serialize_context.image_cache:
                    cmd = [*cmd, "--image-cache", serialize_context.image_cache.to_transport]

            if serialize_context.code_bundle:
                if serialize_context.code_bundle.tgz:
                    cmd = [*cmd, *["--tgz", f"{serialize_context.code_bundle.tgz}"]]
                elif serialize_context.code_bundle.pkl:
                    cmd = [*cmd, *["--pkl", f"{serialize_context.code_bundle.pkl}"]]
                cmd = [*cmd, *["--dest", f"{serialize_context.code_bundle.destination or '.'}"]]

            if self.inputs:
                cmd.append("--inputs")
                cmd.append(self._serialize_inputs())

            return [*cmd, "--"]
        elif isinstance(self.command, str):
            return shlex.split(self.command)
        else:
            # command is a list
            return self.command

    def get_port(self) -> Port:
        if isinstance(self.port, int):
            self.port = Port(port=self.port)
        return self.port

    @property
    def endpoint(self) -> str:
        endpoint_pattern = os.getenv(INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR)
        if endpoint_pattern is not None:
            return endpoint_pattern.replace("{app_fqdn}", self.name)

        import flyte.remote

        app = flyte.remote.App.get(name=self.name)
        return app.endpoint

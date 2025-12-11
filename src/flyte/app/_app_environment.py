from __future__ import annotations

import inspect
import os
import re
import shlex
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Union

import rich.repr

from flyte import Environment, Image, Resources, SecretRequest
from flyte.app._input import Input
from flyte.app._types import Domain, Link, Port, Scaling
from flyte.models import SerializationContext

if TYPE_CHECKING:
    pass


APP_NAME_RE = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")
INVALID_APP_PORTS = [8012, 8022, 8112, 9090, 9091]
INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR = "INTERNAL_APP_ENDPOINT_PATTERN"


@rich.repr.auto
@dataclass(init=True, repr=True)
class AppEnvironment(Environment):
    """
    :param type: Type of the environment.
    :param port: Port to use for the app server.
    :param args: Arguments to pass to app.
    :param command: Command to run in the app.
    :param requires_auth: Whether the app requires authentication.
    :param scaling: Scaling configuration for the app environment.
    :param domain: Domain to use for the app.
    :param links: Links to other environments.
    :param include: Files to include in the environment to run the app.
    :param inputs: Inputs to pass to the app environment.
    :param cluster_pool: Cluster pool to use for the app environment.
    :param name: Name of the app environment
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

    def _get_app_filename(self) -> str:
        """
        Get the filename of the file that declares this app environment.

        Returns the actual file path instead of <string>, skipping flyte SDK internal files.
        """

        def is_user_file(filename: str) -> bool:
            """Check if a file is a user file (not part of flyte SDK)."""
            if filename in ("<string>", "<stdin>"):
                return False
            if not os.path.exists(filename):
                return False
            # Skip files that are part of the flyte SDK
            abs_path = os.path.abspath(filename)
            # Check if file is in flyte package
            return ("site-packages/flyte" not in abs_path and "/flyte/" not in abs_path) or "/examples/" in abs_path

        # Try frame inspection first - walk up the stack to find user file
        frame = inspect.currentframe()
        while frame is not None:
            filename = frame.f_code.co_filename
            if is_user_file(filename):
                return os.path.abspath(filename)
            frame = frame.f_back

        # Fallback: Inspect the full stack to find the first user file
        stack = inspect.stack()
        for frame_info in stack:
            filename = frame_info.filename
            if is_user_file(filename):
                return os.path.abspath(filename)

        # Last fallback: Try to get from __main__ module
        import sys

        if hasattr(sys.modules.get("__main__"), "__file__"):
            main_file = sys.modules["__main__"].__file__
            if main_file and os.path.exists(main_file):
                return os.path.abspath(main_file)

        # Last resort: return the current working directory with a placeholder
        # This shouldn't happen in normal usage
        return os.path.join(os.getcwd(), "app.py")

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
        self._app_filename = self._get_app_filename()

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        if self.args is None:
            return []
        elif isinstance(self.args, str):
            return shlex.split(self.args)
        else:
            # args is a list
            return self.args

    def _serialize_inputs(self, input_overrides: list[Input] | None) -> str:
        if not self.inputs:
            return ""
        from ._input import SerializableInputCollection

        serialized_inputs = SerializableInputCollection.from_inputs(input_overrides or self.inputs)
        return serialized_inputs.to_transport

    def container_cmd(
        self, serialize_context: SerializationContext, input_overrides: list[Input] | None = None
    ) -> List[str]:
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
                cmd.append(self._serialize_inputs(input_overrides))

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
            return endpoint_pattern.format(app_fqdn=self.name)

        import flyte.remote
        from flyte._initialize import ensure_client

        ensure_client()
        app = flyte.remote.App.get(name=self.name)
        return app.endpoint

    def clone_with(
        self,
        name: str,
        image: Optional[Union[str, Image, Literal["auto"]]] = None,
        resources: Optional[Resources] = None,
        env_vars: Optional[dict[str, str]] = None,
        secrets: Optional[SecretRequest] = None,
        depends_on: Optional[List[Environment]] = None,
        description: Optional[str] = None,
        interruptible: Optional[bool] = None,
        **kwargs: Any,
    ) -> AppEnvironment:
        # validate unknown kwargs if needed

        type = kwargs.pop("type", None)
        port = kwargs.pop("port", None)
        args = kwargs.pop("args", None)
        command = kwargs.pop("command", None)
        requires_auth = kwargs.pop("requires_auth", None)
        scaling = kwargs.pop("scaling", None)
        domain = kwargs.pop("domain", None)
        links = kwargs.pop("links", None)
        include = kwargs.pop("include", None)
        inputs = kwargs.pop("inputs", None)
        cluster_pool = kwargs.pop("cluster_pool", None)

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")

        kwargs = self._get_kwargs()
        kwargs["name"] = name
        if image is not None:
            kwargs["image"] = image
        if resources is not None:
            kwargs["resources"] = resources
        if env_vars is not None:
            kwargs["env_vars"] = env_vars
        if secrets is not None:
            kwargs["secrets"] = secrets
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        if description is not None:
            kwargs["description"] = description
        if type is not None:
            kwargs["type"] = type
        if port is not None:
            kwargs["port"] = port
        if args is not None:
            kwargs["args"] = args
        if command is not None:
            kwargs["command"] = command
        if requires_auth is not None:
            kwargs["requires_auth"] = requires_auth
        if scaling is not None:
            kwargs["scaling"] = scaling
        if domain is not None:
            kwargs["domain"] = domain
        if links is not None:
            kwargs["links"] = links
        if include is not None:
            kwargs["include"] = include
        if inputs is not None:
            kwargs["inputs"] = inputs
        if cluster_pool is not None:
            kwargs["cluster_pool"] = cluster_pool
        return replace(self, **kwargs)

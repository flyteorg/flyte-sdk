import copy
import os
import re
import shlex
import warnings
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from inspect import getmembers
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable
from typing import Literal as L

import rich.repr

import flyteidl.core.tasks_pb2 as tasks_pb2
from flyte._environment import Environment
from flyte._image import Image
from flyte._resources import Resources
from flyte._context import ctx
from flyte._pod import PodTemplate
from flyte._secret import Secret
from flyte._internal.runtime.resources_serde import get_proto_resources, get_proto_extended_resources
from flyteidl.core.literals_pb2 import KeyValuePair
from flyteidl.core.tasks_pb2 import Container, ContainerPort, ExtendedResources, K8sObjectMetadata, K8sPod
from mashumaro.codecs.json import JSONEncoder

from flyte.app._common import _extract_files_loaded_from_cwd
from flyte.app._frameworks import _is_fastapi_app
from flyte._protos.app.app_definition_pb2 import App as AppIDL
from flyte._protos.app.app_definition_pb2 import (
    AutoscalingConfig,
    Identifier,
    IngressConfig,
    Meta,
    Replicas,
    SecurityContext,
    Spec,
)
from flyte._protos.app.app_definition_pb2 import Concurrency as ConcurrencyIDL
from flyte._protos.app.app_definition_pb2 import Link as LinkIDL
from flyte._protos.app.app_definition_pb2 import RequestRate as RequestRateIDL
from flyte._protos.app.app_definition_pb2 import ScalingMetric as ScalingMetricIDL


def is_union_image(image: str):
    return image.startswith("cr.union.ai/")


def get_image_name(image_url: str) -> str:
    """
    Extract the image name from a Docker image string.

    :param image_url: Docker image string (e.g., 'repository/name:tag' or 'name:tag')
    :return: The image name
    """
    pattern = r"^(?:[^/]+/)?([^:]+)(?::[^:]+)?$"
    match = re.match(pattern, image_url)
    if not match:
        raise ValueError(f"Invalid Docker image format: {image_url}")
    return match.group(1)


@dataclass
class AppRegistry:
    """Keeps track of all user defined applications"""

    apps: Dict[str, "App"] = field(default_factory=dict)

    def add_app(self, app: "App"):
        context = ctx()
        if (context is None or context.mode is None) and app.name in self.apps:
            # This is running in a non execute context, so we check if the app name is already used:
            msg = f"App with name: {app.name} was already used. The latest definition will be used"
            warnings.warn(msg, UserWarning, stacklevel=2)

        self.apps[app.name] = app


APP_REGISTRY = AppRegistry()
# TODO: Add more protocols here when we deploy to another cloud providers
SUPPORTED_FS_PROTOCOLS = ["s3://", "gs://", "union://", "ufs://", "unionmeta://", "ums://", "abfs://"]
INVALID_APP_PORTS = [8012, 8022, 8112, 9090, 9091]
_PACKAGE_NAME_RE = re.compile(r"^[\w-]+")
_UNION_RUNTIME_NAME = "union-runtime"


def _is_union_runtime(package: str) -> bool:
    """Return True if `package` is union-runtime."""
    m = _PACKAGE_NAME_RE.match(package)
    if not m:
        return False
    name = m.group()
    return name == _UNION_RUNTIME_NAME


def _convert_union_runtime_to_serverless(packages: Optional[list[str]]) -> Optional[list[str]]:
    """Convert packages using union-runtime to union-runtime[serverless]"""
    if packages is None:
        return None

    union_runtime_length = len(_UNION_RUNTIME_NAME)
    new_packages = []
    for package in packages:
        if _is_union_runtime(package):
            version_spec = package[union_runtime_length:]
            new_packages.append(f"union-runtime[serverless]{version_spec}")
        else:
            new_packages.append(package)
    return new_packages


@dataclass
class URLQuery:
    name: str
    public: bool = False


ENV_NAME_RE = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")
APP_NAME_RE = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")


def _has_file_extension(file_path) -> bool:
    _, ext = os.path.splitext(file_path)
    return bool(ext)


@dataclass
class Input:
    """
    Input for application.

    :param name: Name of input.
    :param value: Value for input.
    :param env_var: Environment name to set the value in the serving environment.
    :param download: When True, the input will be automatically downloaded. This
        only works if the value refers to an item in a object store. i.e. `s3://...`
    :param mount: If `value` is a directory, then the directory will be available
        at `mount`. If `value` is a file, then the file will be downloaded into the
        `mount` directory.
    :param ignore_patterns: If `value` is a directory, then this is a list of glob
        patterns to ignore.
    """

    class Type(Enum):
        File = "file"
        Directory = "directory"
        String = "string"
        _UrlQuery = "url_query"  # Private type, users should not need this

    value: Union[str, URLQuery]
    name: Optional[str] = None
    env_var: Optional[str] = None
    type: Optional[Type] = None
    download: bool = False
    mount: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.env_var is not None and ENV_NAME_RE.match(self.env_var) is None:
            msg = f"env_var ({self.env_var}) is not a valid environment name for shells"
            raise ValueError(msg)

        if self.name is None:
            if isinstance(self.value, URLQuery):
                self.name = self.value.name
            elif isinstance(self.value, str):
                self.name = self.value
            else:
                msg = "If name is not provided, then the Input value must be an URLQuery, or str"
                raise ValueError(msg)


@dataclass
class MaterializedInput:
    value: str
    type: Optional[Input.Type] = None


@dataclass
class AppSerializationSettings:
    """Runtime settings for creating an AppIDL"""

    org: str
    project: str
    domain: str
    version: str
    image_uri: str
    is_serverless: bool
    desired_state: Spec.DesiredState
    materialized_inputs: dict[str, MaterializedInput] = field(default_factory=dict)
    additional_distribution: Optional[str] = None


@dataclass
class InputBackend:
    """
    Input information for the backend.
    """

    name: str
    value: str
    download: bool
    type: Optional[Input.Type] = None
    env_var: Optional[str] = None
    dest: Optional[str] = None
    ignore_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.type is None:
            if any(self.value.startswith(proto) for proto in SUPPORTED_FS_PROTOCOLS):
                if _has_file_extension(self.value):
                    self.type = Input.Type.File
                else:
                    self.type = Input.Type.Directory
            else:
                self.type = Input.Type.String

        if self.type == Input.Type.String:
            # Nothing to download for a string
            self.download = False

        # If the type is a file or directory and there is a destination, then we
        # automatically assume it is going to be downloaded
        # TODO: In the future, we may mount this, so there is no need to download it
        # with the runtime.
        if self.type in (Input.Type.File, Input.Type.Directory) and self.dest is not None:
            self.download = True

    @classmethod
    def from_input(cls, user_input: Input, settings: AppSerializationSettings) -> "InputBackend":
        if isinstance(user_input.value, str):
            value = user_input.value
            input_type = user_input.type
        else:
            try:
                materialized_input = settings.materialized_inputs[user_input.name]
                value = materialized_input.value
                input_type = materialized_input.type or user_input.type
            except KeyError:
                msg = f"Did not materialize {user_input.name}"
                raise ValueError(msg)

        return InputBackend(
            name=user_input.name,
            value=value,
            download=user_input.download,
            env_var=user_input.env_var,
            type=input_type,
            dest=user_input.mount,
            ignore_patterns=user_input.ignore_patterns,
        )


@dataclass
class ServeConfig:
    """
    Configuration for serve runtime.

    :param code_uri: Location of user code in an object store (s3://...)
    :param user_inputs: User inputs. Passed in by `app.inputs`
    """

    code_uri: str  # location of user code
    inputs: List[InputBackend]


SERVE_CONFIG_ENCODER = JSONEncoder(ServeConfig)


@dataclass
class ResolvedInclude:
    src: str
    dest: str


class ScalingMetric:
    @dataclass(frozen=True)
    class Concurrency:
        """
        Use this to specify the concurrency metric for autoscaling, i.e. the number of concurrent requests at a replica
         at which to scale up.
        """

        val: int

        def __post_init__(self):
            if self.val < 1:
                raise ValueError("Concurrency must be greater than or equal to 1")

        def _to_union_idl(self) -> ConcurrencyIDL:
            return ConcurrencyIDL(target_value=self.val)

    @dataclass
    class RequestRate:
        """
        Use this to specify the request rate metric for autoscaling, i.e. the number of requests per second at a replica
         at which to scale up.
        """

        val: int

        def __post_init__(self):
            if self.val < 1:
                raise ValueError("Request rate must be greater than or equal to 1")

        def _to_union_idl(self) -> RequestRateIDL:
            return RequestRateIDL(target_value=self.val)

    @staticmethod
    def _to_union_idl(metric: Optional[Union[RequestRate, Concurrency]]) -> Optional[ScalingMetricIDL]:
        if metric is None:
            return None
        request_rate = None
        concurrency = None
        if isinstance(metric, ScalingMetric.RequestRate):
            request_rate = metric._to_union_idl()
        if isinstance(metric, ScalingMetric.Concurrency):
            concurrency = metric._to_union_idl()
        return ScalingMetricIDL(
            request_rate=request_rate,
            concurrency=concurrency,
        )

    @staticmethod
    def _validate(metric: Optional[Union[RequestRate, Concurrency]]):
        if metric is None:
            return
        if not isinstance(metric, (ScalingMetric.RequestRate, ScalingMetric.Concurrency)):
            raise ValueError("scaling_metric must be an instance of Concurrency or RequestRate")


@runtime_checkable
class AppConfigProtocol(Protocol):
    def before_to_union_idl(self, app: "App", settings: AppSerializationSettings):
        """Modify app in place at the beginning of `App._to_union_idl`."""


@dataclass
class Link:
    path: str
    title: str
    is_relative: bool = False


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

    @dataclass
    class Port:
        port: int
        name: Optional[str] = None

    port: Optional[Union[int, Port]] = None
    args: Optional[Union[List[str], str]] = None
    min_replicas: int = 0
    max_replicas: int = 1
    scaledown_after: Optional[Union[int, timedelta]] = None
    scaling_metric: Optional[Union[ScalingMetric.Concurrency, ScalingMetric.RequestRate]] = None
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    cluster_pool: str = "default"
    requires_auth: bool = True
    type: Optional[str] = None
    description: Optional[str] = None
    framework_app: Optional[Any] = None
    config: Optional[AppConfigProtocol] = None
    subdomain: Optional[str] = None
    custom_domain: Optional[str] = None
    links: List[Link] = field(default_factory=list)
    shared_memory: Optional[Union[L[True], str]] = None

    @property
    def app(self) -> "App":
        """
        Return the app in the environment.
        """
        return App(
            name=self.name,
            resources=self.resources,
            env=self.env_vars,
            secrets=self.secrets,
            dependencies=self.depends_on,
            image=self.pod_template or self.image,
            port=self.port,
            args=self.args,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
            scaledown_after=self.scaledown_after,
            scaling_metric=self.scaling_metric,
            include=self.include,
            inputs=self.inputs,
            cluster_pool=self.cluster_pool,
            requires_auth=self.requires_auth,
            type=self.type,
            description=self.description,
            framework_app=self.framework_app,
            config=self.config,
            subdomain=self.subdomain,
            custom_domain=self.custom_domain,
            links=self.links,
            shared_memory=self.shared_memory,
        )


@dataclass
class App:
    """
    App specification.

    :param name: The name of the application.
    :param image: The container image to use for the application.
    :param args: Entrypoint to start application.
    :param command: Command to start application.
    :param port: Port application listens to. Currently, this must be 8080 and the application
        must listen on 8080.
    :param requests: Compute resource requests for application.
    :param secrets: Secrets that are requested for application.
    :param limits: Compute resource limits for application.
    :param include: Files to include for your application.
    :param inputs: Inputs for the application.
    :param env: Environment variables to set for the application.
    :param min_replicas: Minimum number of replicas (ignore if autoscaling is set).
    :param max_replicas: Maximum number of replicas (ignore if autoscaling is set).
    :param scaledown_after: Time to wait before scaling down a replica after it has been idle.
    :param scaling_metric: Autoscale based on a parameter, e.g. request rate or concurrency
      (others may be added in the future).
    :param cluster_pool: The target cluster_pool where the app should be deployed.
    :param requires_auth: Public URL does not require any authentication
    :param type: Type of app
    :param description: Description of app
    :param framework_app: Object for serving framework. When this is set, all user defined files are uploaded.
        - For FastAPI, args is set to `uvicorn module_name:app_name --port port`. For more control
          you can set `args` directly.
    :param dependencies: List of apps that this app depends on.
    :param subdomain: Custom subdomain for your app.
    :param custom_domain: Custom full domain for your app.
    :param links: Links to external URLs or relative paths.
    :param shared_memory: If True, then shared memory will be attached to the container where the size is equal
            to the allocated memory. If str, then the shared memory is set to that size.
    """

    @dataclass
    class Port:
        port: int
        name: Optional[str] = None

    name: str
    image: Union[str, Image, PodTemplate]
    port: Optional[Union[int, Port]] = None
    resources: Optional[Resources] = None
    secrets: List[Secret] = field(default_factory=list)
    args: Optional[Union[List[str], str]] = None
    command: Optional[Union[List[str], str]] = None
    min_replicas: int = 0
    max_replicas: int = 1
    scaledown_after: Optional[Union[int, timedelta]] = None
    scaling_metric: Optional[Union[ScalingMetric.Concurrency, ScalingMetric.RequestRate]] = None
    include: List[str] = field(default_factory=list)
    inputs: List[Input] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    cluster_pool: str = "default"
    requires_auth: bool = True
    type: Optional[str] = None
    description: Optional[str] = None
    framework_app: Optional[Any] = None
    dependencies: List["App"] = field(default_factory=list)
    config: Optional[AppConfigProtocol] = None
    subdomain: Optional[str] = None
    custom_domain: Optional[str] = None
    links: List[Link] = field(default_factory=list)
    shared_memory: Optional[Union[L[True], str]] = None

    _include_resolved: Optional[List[ResolvedInclude]] = field(default=None, init=False)
    _port: Optional[Port] = field(default=None, init=False)
    _framework_variable_name: Optional[str] = field(default=None, init=False)
    _module_name: Optional[str] = field(default=None, init=False)

    def _validate_autoscaling_config(self):
        if self.min_replicas < 0:
            raise ValueError("min_replicas must be greater than or equal to 0")
        if self.max_replicas < 1 or self.max_replicas < self.min_replicas:
            raise ValueError(f"max_replicas must be greater than or equal to 1 or min_replicas:{self.min_replicas}")
        if self.scaledown_after is not None:
            if isinstance(self.scaledown_after, int):
                self.scaledown_after = timedelta(seconds=self.scaledown_after)
            if not 6 <= self.scaledown_after.total_seconds() <= 3600:
                raise ValueError("scaledown_after must be between 6 seconds and 1 hour")
        ScalingMetric._validate(self.scaling_metric)

    def _validate_ports(self):
        if isinstance(self.port, int):
            self._port = App.Port(port=self.port)
        elif isinstance(self.port, App.Port):
            self._port = self.port
        else:
            msg = "Expected port to be int, list[int], App.Port, or list[App.Port]"
            raise ValueError(msg)

        if self._port.port in INVALID_APP_PORTS:
            invalid_ports = ", ".join(str(p) for p in INVALID_APP_PORTS)
            msg = f"port {self._port.port} is not allowed. Please do not use ports: {invalid_ports}"
            raise ValueError(msg)

    def __post_init__(self):
        self._validate_autoscaling_config()

        if self.inputs and self.command is not None:
            msg = "To use inputs, set command=None to use the union-serve entrypoint"
            raise ValueError(msg)

        if self.include and self.command is not None:
            msg = "To use include, set command=None to use th union-serve entrypoint"
            raise ValueError(msg)

        if not self.framework_app:
            if self.args is None and self.command is None:
                msg = "Provide args or command to start your app. If you have default entrypoint then set command=[]"
                raise ValueError(msg)

            if self.port is None:
                msg = "Provide a port to connect to"
                raise ValueError(msg)
        else:
            if self.command is not None:
                msg = "When using framework_app, set command=None to use the union-runtime endpoint"
                raise ValueError(msg)

            if (
                self.type is None
                and hasattr(self.framework_app, "__class__")
                and hasattr(self.framework_app.__class__, "__name__")
            ):
                self.type = self.framework_app.__class__.__name__

            if self.port is None:
                self.port = 8080

        self._validate_ports()

        match_re = APP_NAME_RE.fullmatch(self.name)
        if not match_re:
            raise ValueError("App name must consist of lower case alphanumeric characters, '-' or '.'")

        if self.resources and self.resources.disk is None:
            self.resources.disk = "20Gi"

        APP_REGISTRY.add_app(self)

    @property
    def include_resolved(self) -> List[ResolvedInclude]:
        # Already resolved
        if self._include_resolved is not None:
            return self._include_resolved

        # Not resolved yet, so we resolve based on the cwd. This only happens when using
        # AppRemote programmatically and not from `union deploy`
        self._resolve_include(Path.cwd(), Path.cwd())
        return self._include_resolved

    def _attach_registration_scope(self, module: Optional[ModuleType], module_name: Optional[str]) -> "App":
        """
        Attach variable name to the object
        """
        if self.framework_app:
            # extract variable name from module
            for var_name, obj in getmembers(module):
                if obj is self.framework_app:
                    self._framework_variable_name = var_name
                    break
            else:  # no break
                msg = "Unable to find framework_app in the scope of App"
                raise RuntimeError(msg)
        self._module_name = module_name
        return self

    def _resolve_include(self, app_directory: Path, cwd: Path) -> "App":
        """
        Resolve include based on the working_dir.

        If a path in `include` is prefixed with "./", then those files are
        assumed to be relative to the file that has the App object.
        """
        relative_prefix = "./"
        seen_dests = set()

        included_resolved = []

        all_includes = self.include

        if self.framework_app is not None:
            all_includes.extend(_extract_files_loaded_from_cwd(cwd))

        for file in all_includes:
            normed_file = os.path.normpath(file)
            if file.startswith(relative_prefix):
                # File is relative to the app_directory:
                src_dir = app_directory
            else:
                src_dir = cwd

            if "*" in normed_file:
                new_srcs = src_dir.rglob(normed_file)
            else:
                new_srcs = [src_dir / normed_file]

            for new_src in new_srcs:
                dest = new_src.relative_to(src_dir).as_posix()
                if dest in seen_dests:
                    msg = f"{dest} is in include multiple times. Please remove one of them."
                    raise ValueError(msg)

                seen_dests.add(dest)

                included_resolved.append(ResolvedInclude(src=os.fspath(new_src), dest=os.fspath(dest)))

        self._include_resolved = included_resolved
        return self

    def _get_image(self, container_image: Union[str, Image], settings: AppSerializationSettings) -> str:
        if isinstance(container_image, str):
            if settings.is_serverless and not is_union_image(container_image):
                # Serverless expects the image to be built by the serverless runtime.
                # If the image isn't a union image, then we need to build it.
                container_image = Image(name=get_image_name(container_image), base_image=container_image)
            else:
                return container_image

        image_spec = container_image

        if settings.is_serverless and image_spec.packages is not None:
            image_spec = deepcopy(image_spec)
            new_packages = _convert_union_runtime_to_serverless(image_spec.packages)
            image_spec.packages = new_packages

        return self._build_image(image_spec)

    def _build_image(self, image: Image):
        from flyte._build import build

        ctx = nullcontext()
        with ctx:
            build(image)
        return image.name

    def _construct_args_for_framework(self, framework_app: Any) -> Optional[str]:
        # Framework specific argument adjustments here
        if _is_fastapi_app(framework_app):
            if self._module_name is None or self._framework_variable_name is None:
                raise ValueError("Unable to find module name")
            return f"uvicorn {self._module_name}:{self._framework_variable_name} --port {self.port}"

        return None

    def _get_args(self) -> List[str]:
        args = self.args

        # Framework specific argument adjustments here
        if self.framework_app is not None and args is None:
            args = self._construct_args_for_framework(self.framework_app)

        if args is None:
            return []
        elif isinstance(args, str):
            return shlex.split(args)
        else:
            # args is a list
            return args

    def _get_command(self, settings: AppSerializationSettings) -> List[str]:
        if self.command is None:
            cmd = ["union-serve"]

            serve_config = ServeConfig(
                code_uri=settings.additional_distribution,
                inputs=[InputBackend.from_input(user_input, settings) for user_input in self.inputs],
            )

            cmd.extend(["--config", SERVE_CONFIG_ENCODER.encode(serve_config)])

            return [*cmd, "--"]
        elif isinstance(self.command, str):
            return shlex.split(self.command)
        else:
            # args is a list
            return self.command

    def _get_resources(self) -> tasks_pb2.Resources:
        return get_proto_resources(self.resources)

    def _get_container(self, settings: AppSerializationSettings) -> Container:
        container_ports = [ContainerPort(container_port=self._port.port, name=self._port.name)]

        return Container(
            image=self._get_image(self.image, settings),
            command=self._get_command(settings),
            args=self._get_args(),
            resources=self._get_resources(),
            ports=container_ports,
            env=[KeyValuePair(key=k, value=v) for k, v in self._get_env(settings).items()],
        )

    def _get_env(self, settings: AppSerializationSettings) -> dict:
        return self.env or {}

    def _get_extended_resources(self) -> Optional[ExtendedResources]:
        return get_proto_extended_resources(self.resources)

    def _to_union_idl(self, settings: AppSerializationSettings) -> AppIDL:
        if self.config is not None:
            self.config.before_to_union_idl(self, settings)

        security_context_kwargs = {}
        security_context = None
        if self.secrets:
            security_context_kwargs["secrets"] = [s.to_flyte_idl() for s in self.secrets]
        if not self.requires_auth:
            security_context_kwargs["allow_anonymous"] = True

        if security_context_kwargs:
            security_context = SecurityContext(**security_context_kwargs)

        scaling_metric = ScalingMetric._to_union_idl(self.scaling_metric)

        dur = None
        if self.scaledown_after:
            from google.protobuf.duration_pb2 import Duration

            dur = Duration()
            dur.FromTimedelta(self.scaledown_after)

        autoscaling = AutoscalingConfig(
            replicas=Replicas(min=self.min_replicas, max=self.max_replicas),
            scaledown_period=dur,
            scaling_metric=scaling_metric,
        )

        spec_kwargs = {}
        if isinstance(self.image, (str, Image)):
            spec_kwargs["container"] = self._get_container(settings)
        else:
            msg = "container_image must be a str, ImageSpec or PodTemplate"
            raise ValueError(msg)

        from flyte._protos.app.app_definition_pb2 import Profile

        return AppIDL(
            metadata=Meta(
                id=Identifier(
                    org=settings.org,
                    project=settings.project,
                    domain=settings.domain,
                    name=self.name,
                ),
            ),
            spec=Spec(
                desired_state=settings.desired_state,
                ingress=IngressConfig(
                    private=False,
                    subdomain=self.subdomain if self.subdomain else None,
                    cname=self.custom_domain if self.custom_domain else None,
                ),
                autoscaling=autoscaling,
                security_context=security_context,
                cluster_pool=self.cluster_pool,
                extended_resources=self._get_extended_resources(),
                profile=Profile(
                    type=self.type,
                    short_description=self.description,
                ),
                links=[LinkIDL(path=link.path, title=link.title, is_relative=link.is_relative) for link in self.links]
                if self.links
                else None,
                **spec_kwargs,
            ),
        )

    @classmethod
    def _update_app_idl(cls, old_app_idl: AppIDL, new_app_idl: AppIDL) -> AppIDL:
        # Replace all lists with empty so that MergeFrom works out of the box.
        app_idl_ = AppIDL(
            metadata=old_app_idl.metadata,
            spec=new_app_idl.spec,
            status=old_app_idl.status,
        )
        # Make sure values set by the server and not by the app configuration is
        # preserved.
        if old_app_idl.spec.creator.ListFields():
            app_idl_.spec.creator.CopyFrom(old_app_idl.spec.creator)

        # Ingress subdomain could be configured by the server or overriden by the user
        app_idl_.spec.ingress.CopyFrom(old_app_idl.spec.ingress)
        app_idl_.spec.ingress.MergeFrom(new_app_idl.spec.ingress)
        return app_idl_

    def _get_k8s_pod(self, pod_template: PodTemplate, settings: AppSerializationSettings) -> K8sPod:
        """Convert pod_template into a K8sPod IDL."""
        import json

        from google.protobuf.json_format import Parse
        from google.protobuf.struct_pb2 import Struct

        pod_spec_dict = self._serialized_pod_spec(pod_template, settings)
        pod_spec_idl = Parse(json.dumps(pod_spec_dict), Struct())

        metadata = K8sObjectMetadata(
            labels=pod_template.labels,
            annotations=pod_template.annotations,
        )
        return K8sPod(pod_spec=pod_spec_idl, metadata=metadata)

    @staticmethod
    def _sanitize_resource_name(resource: tasks_pb2.Resources.ResourceEntry) -> str:
        return tasks_pb2.Resources.ResourceName.Name(resource.name).lower().replace("_", "-")

    def _serialized_pod_spec(
        self,
        pod_template: PodTemplate,
        settings: AppSerializationSettings,
    ) -> dict:
        """Convert pod spec into a dict."""
        from kubernetes.client import ApiClient
        from kubernetes.client.models import V1Container, V1ContainerPort, V1EnvVar, V1ResourceRequirements

        pod_template = copy.deepcopy(pod_template)

        if pod_template.pod_spec is None:
            return {}

        if pod_template.primary_container_name != "app":
            msg = "Primary container name must be 'app'"
            raise ValueError(msg)

        containers: list[V1Container] = pod_template.pod_spec.containers
        primary_exists = any(container.name == pod_template.primary_container_name for container in containers)

        if not primary_exists:
            msg = "Primary container does not exist with name 'app' does not exist"
            raise ValueError(msg)

        final_containers = []

        # Move all resource names into containers
        for container in containers:
            container.image = self._get_image(container.image, settings)

            if container.name == pod_template.primary_container_name:
                container.args = self._get_args()
                container.command = self._get_command(settings)

                limits, requests = {}, {}
                resources = self._get_resources()
                for resource in resources.limits:
                    limits[self._sanitize_resource_name(resource)] = resource.value
                for resource in resources.requests:
                    requests[self._sanitize_resource_name(resource)] = resource.value

                resource_requirements = V1ResourceRequirements(limits=limits, requests=requests)

                if limits or requests:
                    container.resources = resource_requirements

                if self.env:
                    container.env = [V1EnvVar(name=k, value=v) for k, v in self.env.items()] + (container.env or [])

                container.ports = [V1ContainerPort(container_port=self._port.port, name=self._port.name)] + (
                    container.ports or []
                )

            final_containers.append(container)

        pod_template.pod_spec.containers = final_containers
        return ApiClient().sanitize_for_serialization(pod_template.pod_spec)

    def query_endpoint(self, *, public: bool = False) -> URLQuery:
        """
        Query for endpoint.

        :param public: Whether to return the public or internal endpoint.
        :returns: Object representing a URL query.
        """
        raise NotImplementedError
        return URLQuery(name=self.name, public=public)

    @property
    def endpoint(self) -> str:
        """
        Return endpoint for App.
        """
        raise NotImplementedError
        pattern = os.getenv(INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR)
        if pattern is not None:
            # If the pattern exist, we are on remote.
            # NOTE: Should check that the app is in the same namespace as the current app.
            return pattern.replace("{app_fqdn}", self.name)

        from union.remote import UnionRemote

        remote = UnionRemote()
        app_remote = remote._app_remote
        app_idl = app_remote.get(name=self.name)
        url = app_idl.status.ingress.public_url
        return url

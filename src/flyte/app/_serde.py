from flyte._internal.runtime.resources_serde import get_proto_resources, get_proto_extended_resources
from flyte._protos.app.app_definition_pb2 import App as AppIDL, SecurityContext, AutoscalingConfig, Replicas, Meta, \
    Identifier, Spec, IngressConfig, Link as LinkIDL
from flyteidl.core import tasks_pb2, literals_pb2

from flyte.app._common import _extract_files_loaded_from_cwd
from flyte.app._frameworks import _is_fastapi_app

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


def _get_container(self, settings: AppSerializationSettings) -> tasks_pb2.Container:
    container_ports = [tasks_pb2.ContainerPort(container_port=self._port.port, name=self._port.name)]

    return tasks_pb2.Container(
        image=self._get_image(self.image, settings),
        command=self._get_command(settings),
        args=self._get_args(),
        resources=self._get_resources(),
        ports=container_ports,
        env=[literals_pb2.KeyValuePair(key=k, value=v) for k, v in self._get_env(settings).items()],
    )


def _get_env(self, settings: AppSerializationSettings) -> dict:
    return self.env or {}


def _get_extended_resources(self) -> Optional[tasks_pb2.ExtendedResources]:
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


def _get_k8s_pod(self, pod_template: PodTemplate, settings: AppSerializationSettings) -> tasks_pb2.K8sPod:
    """Convert pod_template into a K8sPod IDL."""
    import json

    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Struct

    pod_spec_dict = self._serialized_pod_spec(pod_template, settings)
    pod_spec_idl = Parse(json.dumps(pod_spec_dict), Struct())

    metadata = tasks_pb2.K8sObjectMetadata(
        labels=pod_template.labels,
        annotations=pod_template.annotations,
    )
    return tasks_pb2.K8sPod(pod_spec=pod_spec_idl, metadata=metadata)


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
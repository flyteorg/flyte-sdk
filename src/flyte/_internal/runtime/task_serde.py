"""
This module provides functionality to serialize and deserialize tasks to and from the wire format.
It includes a Resolver interface for loading tasks, and functions to load classes and tasks.
"""

import copy
import typing
from datetime import timedelta
from typing import Optional, cast

from flyteidl2.core import identifier_pb2, literals_pb2, security_pb2, tasks_pb2
from flyteidl2.task import common_pb2, environment_pb2, task_definition_pb2
from google.protobuf import duration_pb2, wrappers_pb2

import flyte.errors
from flyte._cache.cache import VersionParameters, cache_from_request
from flyte._logging import logger
from flyte._pod import _PRIMARY_CONTAINER_NAME_FIELD, PodTemplate
from flyte._secret import SecretRequest, secrets_from_request
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import CodeBundle, SerializationContext

from ... import ReusePolicy
from ..._retry import RetryStrategy
from ..._timeout import TimeoutType, timeout_from_request
from .resources_serde import get_proto_extended_resources, get_proto_resources
from .reuse import add_reusable
from .types_serde import transform_native_to_typed_interface

_MAX_ENV_NAME_LENGTH = 63  # Maximum length for environment names
_MAX_TASK_SHORT_NAME_LENGTH = 63  # Maximum length for task short names


def translate_task_to_wire(
    task: TaskTemplate,
    serialization_context: SerializationContext,
    default_inputs: Optional[typing.List[common_pb2.NamedParameter]] = None,
) -> task_definition_pb2.TaskSpec:
    """
    Translate a task to a wire format. This is a placeholder function.

    :param task: The task to translate.
    :param serialization_context: The serialization context to use for the translation.
    :param default_inputs: Optional list of default inputs for the task.

    :return: The translated task.
    """
    tt = get_proto_task(task, serialization_context)
    env: environment_pb2.Environment | None = None
    if task.parent_env and task.parent_env():
        _env = task.parent_env()
        if _env:
            env = environment_pb2.Environment(name=_env.name[:_MAX_ENV_NAME_LENGTH])
    return task_definition_pb2.TaskSpec(
        task_template=tt,
        default_inputs=default_inputs,
        short_name=task.short_name[:_MAX_TASK_SHORT_NAME_LENGTH],
        environment=env,
    )


def get_security_context(
    secrets: Optional[SecretRequest],
) -> Optional[security_pb2.SecurityContext]:
    """
    Get the security context from a list of secrets. This is a placeholder function.

    :param secrets: The list of secrets to use for the security context.

    :return: The security context.
    """
    if secrets is None:
        return None

    secret_list = secrets_from_request(secrets)
    return security_pb2.SecurityContext(
        secrets=[
            security_pb2.Secret(
                group=secret.group,
                key=secret.key,
                mount_requirement=(
                    security_pb2.Secret.MountType.ENV_VAR if secret.as_env_var else security_pb2.Secret.MountType.FILE
                ),
                env_var=secret.as_env_var,
            )
            for secret in secret_list
        ]
    )


def get_proto_retry_strategy(
    retries: RetryStrategy | int | None,
) -> Optional[literals_pb2.RetryStrategy]:
    if retries is None:
        return None

    if isinstance(retries, int):
        raise AssertionError("Retries should be an instance of RetryStrategy, not int")

    return literals_pb2.RetryStrategy(retries=retries.count)


def get_proto_timeout(timeout: TimeoutType | None) -> Optional[duration_pb2.Duration]:
    if timeout is None:
        return None
    max_runtime_timeout = timeout_from_request(timeout).max_runtime
    if isinstance(max_runtime_timeout, int):
        max_runtime_timeout = timedelta(seconds=max_runtime_timeout)
    return duration_pb2.Duration(seconds=max_runtime_timeout.seconds)


def get_proto_task(task: TaskTemplate, serialize_context: SerializationContext) -> tasks_pb2.TaskTemplate:
    task_id = identifier_pb2.Identifier(
        resource_type=identifier_pb2.ResourceType.TASK,
        project=serialize_context.project,
        domain=serialize_context.domain,
        org=serialize_context.org,
        name=task.name,
        version=serialize_context.version,
    )

    # TODO Add support for extra_config, custom
    extra_config: typing.Dict[str, str] = {}

    if task.pod_template and not isinstance(task.pod_template, str):
        pod = _get_k8s_pod(_get_urun_container(serialize_context, task), task.pod_template)
        extra_config[_PRIMARY_CONTAINER_NAME_FIELD] = task.pod_template.primary_container_name
        container = None
    else:
        container = _get_urun_container(serialize_context, task)
        pod = None

    custom = task.custom_config(serialize_context)

    sql = task.sql(serialize_context)

    # -------------- CACHE HANDLING ----------------------
    task_cache = cache_from_request(task.cache)
    cache_enabled = task_cache.is_enabled()
    cache_version = None

    if task_cache.is_enabled():
        logger.debug(f"Cache enabled for task {task.name}")
        if serialize_context.code_bundle and serialize_context.code_bundle.pkl:
            logger.debug(f"Detected pkl bundle for task {task.name}, using computed version as cache version")
            cache_version = serialize_context.code_bundle.computed_version
        else:
            if isinstance(task, AsyncFunctionTaskTemplate):
                version_parameters = VersionParameters(func=task.func, image=task.image)
            else:
                version_parameters = VersionParameters(func=None, image=task.image)
            cache_version = task_cache.get_version(version_parameters)
            logger.debug(f"Cache version for task {task.name} is {cache_version}")
    else:
        logger.debug(f"Cache disabled for task {task.name}")

    task_template = tasks_pb2.TaskTemplate(
        id=task_id,
        type=task.task_type,
        metadata=tasks_pb2.TaskMetadata(
            discoverable=cache_enabled,
            discovery_version=cache_version,
            cache_serializable=task_cache.serialize,
            cache_ignore_input_vars=(task_cache.get_ignored_inputs() if cache_enabled else None),
            runtime=tasks_pb2.RuntimeMetadata(
                version=flyte.version(),
                type=tasks_pb2.RuntimeMetadata.RuntimeType.FLYTE_SDK,
                flavor="python",
            ),
            retries=get_proto_retry_strategy(task.retries),
            timeout=get_proto_timeout(task.timeout),
            pod_template_name=(task.pod_template if task.pod_template and isinstance(task.pod_template, str) else None),
            interruptible=task.interruptible,
            generates_deck=wrappers_pb2.BoolValue(value=task.report),
            debuggable=task.debuggable,
        ),
        interface=transform_native_to_typed_interface(task.native_interface),
        custom=custom if len(custom) > 0 else None,
        container=container,
        task_type_version=task.task_type_version,
        security_context=get_security_context(task.secrets),
        config=extra_config,
        k8s_pod=pod,
        sql=sql,
        extended_resources=get_proto_extended_resources(task.resources),
    )

    if task.reusable is not None:
        if not isinstance(task.reusable, ReusePolicy):
            raise flyte.errors.RuntimeUserError(
                "BadConfig", f"Expected ReusePolicy, got {type(task.reusable)} for task {task.name}"
            )
        env_name = None
        if task.parent_env is not None:
            env = task.parent_env()
            if env is not None:
                env_name = env.name
        return add_reusable(task_template, task.reusable, serialize_context.code_bundle, env_name)

    return task_template


def lookup_image_in_cache(serialize_context: SerializationContext, env_name: str, image: flyte.Image) -> str:
    if not serialize_context.image_cache:
        # This computes the image uri, computing hashes as necessary so can fail if done remotely.
        return image.uri
    elif serialize_context.image_cache and env_name not in serialize_context.image_cache.image_lookup:
        raise flyte.errors.RuntimeUserError(
            "MissingEnvironment",
            f"Environment '{env_name}' not found in image cache.\n\n"
            "💡 To fix this:\n"
            "  1. If your parent environment calls a task in another environment,"
            " declare that dependency using 'depends_on=[...]'.\n"
            "     Example:\n"
            "         env1 = flyte.TaskEnvironment(\n"
            "             name='outer',\n"
            "             image=flyte.Image.from_debian_base().with_pip_packages('requests'),\n"
            "             depends_on=[env2, env3],\n"
            "         )\n"
            "  2. If you're using os.getenv() to set the environment name,"
            " make sure the runtime environment has the same environment variable defined.\n"
            "     Example:\n"
            "         env = flyte.TaskEnvironment(\n"
            '             name=os.getenv("my-name"),\n'
            '             env_vars={"my-name": os.getenv("my-name")},\n'
            "         )\n",
        )
    return serialize_context.image_cache.image_lookup[env_name]


def _get_urun_container(
    serialize_context: SerializationContext, task_template: TaskTemplate
) -> Optional[tasks_pb2.Container]:
    env = (
        [literals_pb2.KeyValuePair(key=k, value=v) for k, v in task_template.env_vars.items()]
        if task_template.env_vars
        else None
    )
    resources = get_proto_resources(task_template.resources)

    img = task_template.image
    if isinstance(img, str):
        raise flyte.errors.RuntimeSystemError("BadConfig", "Image is not a valid image")

    env_name = task_template.parent_env_name
    if env_name is None:
        raise flyte.errors.RuntimeSystemError("BadConfig", f"Task {task_template.name} has no parent environment name")

    img_uri = lookup_image_in_cache(serialize_context, env_name, img)

    return tasks_pb2.Container(
        image=img_uri,
        command=[],
        args=task_template.container_args(serialize_context),
        resources=resources,
        env=env,
        data_config=task_template.data_loading_config(serialize_context),
        config=task_template.config(serialize_context),
    )


def _sanitize_resource_name(resource: tasks_pb2.Resources.ResourceEntry) -> str:
    return tasks_pb2.Resources.ResourceName.Name(resource.name).lower().replace("_", "-")


def _get_k8s_pod(primary_container: tasks_pb2.Container, pod_template: PodTemplate) -> Optional[tasks_pb2.K8sPod]:
    """
    Get the K8sPod representation of the task template.
    :param task: The task to convert.
    :return: The K8sPod representation of the task template.
    """
    from kubernetes.client import ApiClient, V1PodSpec
    from kubernetes.client.models import V1EnvVar, V1ResourceRequirements

    pod_template = copy.deepcopy(pod_template)
    containers = cast(V1PodSpec, pod_template.pod_spec).containers
    primary_exists = False

    for container in containers:
        if container.name == pod_template.primary_container_name:
            primary_exists = True
            break

    if not primary_exists:
        raise ValueError(
            "No primary container defined in the pod spec."
            f" You must define a primary container with the name '{pod_template.primary_container_name}'."
        )
    final_containers = []

    for container in containers:
        # We overwrite the primary container attributes with the values given to ContainerTask.
        # The attributes include: image, command, args, resource, and env (env is unioned)

        if container.name == pod_template.primary_container_name:
            if container.image is None:
                # Copy the image from primary_container only if the image is not specified in the pod spec.
                container.image = primary_container.image

            container.command = list(primary_container.command)
            container.args = list(primary_container.args)

            limits, requests = {}, {}
            for resource in primary_container.resources.limits:
                limits[_sanitize_resource_name(resource)] = resource.value
            for resource in primary_container.resources.requests:
                requests[_sanitize_resource_name(resource)] = resource.value

            resource_requirements = V1ResourceRequirements(limits=limits, requests=requests)
            if len(limits) > 0 or len(requests) > 0:
                # Important! Only copy over resource requirements if they are non-empty.
                container.resources = resource_requirements

            if primary_container.env is not None:
                container.env = [V1EnvVar(name=e.key, value=e.value) for e in primary_container.env] + (
                    container.env or []
                )

        final_containers.append(container)

    cast(V1PodSpec, pod_template.pod_spec).containers = final_containers
    pod_spec = ApiClient().sanitize_for_serialization(pod_template.pod_spec)

    metadata = tasks_pb2.K8sObjectMetadata(labels=pod_template.labels, annotations=pod_template.annotations)
    return tasks_pb2.K8sPod(pod_spec=pod_spec, metadata=metadata)


def extract_code_bundle(
    task_spec: task_definition_pb2.TaskSpec,
) -> Optional[CodeBundle]:
    """
    Extract the code bundle from the task spec.
    :param task_spec: The task spec to extract the code bundle from.
    :return: The extracted code bundle or None if not present.
    """
    container = task_spec.task_template.container
    if container and container.args:
        pkl_path = None
        tgz_path = None
        dest_path: str = "."
        version = ""
        for i, v in enumerate(container.args):
            if v == "--pkl":
                # Extract the code bundle path from the argument
                pkl_path = container.args[i + 1] if i + 1 < len(container.args) else None
            elif v == "--tgz":
                # Extract the code bundle path from the argument
                tgz_path = container.args[i + 1] if i + 1 < len(container.args) else None
            elif v == "--dest":
                # Extract the destination path from the argument
                dest_path = container.args[i + 1] if i + 1 < len(container.args) else "."
            elif v == "--version":
                # Extract the version from the argument
                version = container.args[i + 1] if i + 1 < len(container.args) else ""
        if pkl_path or tgz_path:
            return CodeBundle(
                destination=dest_path,
                tgz=tgz_path,
                pkl=pkl_path,
                computed_version=version,
            )
    return None

"""
Serialization module for AppEnvironment to AppIDL conversion.

This module provides functionality to serialize an AppEnvironment object into
the AppIDL protobuf format, using SerializationContext for configuration.
"""
from __future__ import annotations

import os
import shlex
import tarfile
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, List, Optional, Union

from flyteidl2.common import runtime_version_pb2
from flyteidl2.core import literals_pb2, tasks_pb2
from google.protobuf.duration_pb2 import Duration

from flyte import Image, PodTemplate
from flyte._internal.runtime.resources_serde import get_proto_extended_resources, get_proto_resources
from flyte._protos.app import app_definition_pb2
from flyte.app._frameworks import _is_fastapi_app
from flyte.models import SerializationContext

if TYPE_CHECKING:
    from dataclasses import dataclass

    from flyte.app._app_environment import AppEnvironment

    @dataclass
    class ResolvedInclude:
        """Resolved include file with source and destination paths."""
        src: str
        dest: str

FILES_TAR_FILE_NAME = "include-files.tar.gz"


async def upload_include_files(app_env: AppEnvironment, include_resolved: List[ResolvedInclude]) -> str | None:
    """
    Upload include files to remote storage.

    Args:
        app_env: The app environment
        include_resolved: List of resolved include files

    Returns:
        URL to the uploaded tar file, or None if no files to upload
    """
    import flyte.remote

    if not include_resolved:
        return None

    with TemporaryDirectory() as temp_dir:
        tar_path = os.path.join(temp_dir, FILES_TAR_FILE_NAME)
        with tarfile.open(tar_path, "w:gz") as tar:
            for resolve_include in include_resolved:
                tar.add(resolve_include.src, arcname=resolve_include.dest)

        _, upload_native_url = await flyte.remote.upload_file.aio(Path(tar_path))
        return upload_native_url


def _get_image_uri(image: Union[str, Image], serialization_context: SerializationContext) -> str:
    """
    Get the image URI from the image specification.

    Args:
        image: Either a string image name or an Image object
        serialization_context: Serialization context with image cache

    Returns:
        The image URI string
    """
    if isinstance(image, str):
        return image

    # Image object - look up in the image cache
    if image.identifier not in serialization_context.image_cache.image_lookup:
        raise ValueError(f"Image {image.identifier} not found in image cache")

    return serialization_context.image_cache.image_lookup[image.identifier]


def _construct_args_for_framework(
    app_env: AppEnvironment,
    framework_app: Any,
    module_name: Optional[str],
    framework_variable_name: Optional[str],
    port: int
) -> Optional[str]:
    """
    Construct framework-specific arguments.

    Args:
        app_env: The app environment
        framework_app: The framework application object
        module_name: Name of the module containing the app
        framework_variable_name: Variable name of the framework app
        port: Port number

    Returns:
        Command string for the framework, or None
    """
    if _is_fastapi_app(framework_app):
        if module_name is None or framework_variable_name is None:
            raise ValueError("Unable to find module name or framework variable name")
        return f"uvicorn {module_name}:{framework_variable_name} --port {port}"

    return None


def _get_args(
    app_env: AppEnvironment,
    module_name: Optional[str],
    framework_variable_name: Optional[str],
    port: int
) -> List[str]:
    """
    Get the command arguments for the app.

    Args:
        app_env: The app environment
        module_name: Name of the module containing the app
        framework_variable_name: Variable name of the framework app
        port: Port number

    Returns:
        List of argument strings
    """
    args = app_env.args

    # Framework specific argument adjustments
    if app_env.framework_app is not None and args is None:
        args = _construct_args_for_framework(
            app_env,
            app_env.framework_app,
            module_name,
            framework_variable_name,
            port
        )

    if args is None:
        return []
    elif isinstance(args, str):
        return shlex.split(args)
    else:
        # args is a list
        return args


def _get_command(app_env: AppEnvironment, additional_distribution: Optional[str]) -> List[str]:
    """
    Get the command for the app.

    Args:
        app_env: The app environment
        additional_distribution: URL to additional distribution files

    Returns:
        List of command strings
    """
    if app_env.command is None:
        # Default command
        cmd = ["union-serve"]

        if additional_distribution:
            # TODO: Add serve config construction when needed
            # For now, just pass the distribution
            pass

        return [*cmd, "--"]
    elif isinstance(app_env.command, str):
        return shlex.split(app_env.command)
    else:
        # command is a list
        return app_env.command


def _get_env(app_env: AppEnvironment) -> dict:
    """
    Get environment variables for the app.

    Args:
        app_env: The app environment

    Returns:
        Dictionary of environment variables
    """
    return app_env.env_vars or {}


def _get_container(
    app_env: AppEnvironment,
    serialization_context: SerializationContext,
    additional_distribution: Optional[str],
    port: int,
    module_name: Optional[str],
    framework_variable_name: Optional[str],
) -> tasks_pb2.Container:
    """
    Construct the container specification.

    Args:
        app_env: The app environment
        serialization_context: Serialization context
        additional_distribution: URL to additional distribution files
        port: Port number
        module_name: Name of the module containing the app
        framework_variable_name: Variable name of the framework app

    Returns:
        Container protobuf message
    """
    from flyte.app._types import Port

    # Default port if not specified
    _port = Port(port=port, name="http")
    container_ports = [tasks_pb2.ContainerPort(container_port=_port.port, name=_port.name)]

    return tasks_pb2.Container(
        image=_get_image_uri(app_env.image, serialization_context),
        command=_get_command(app_env, additional_distribution),
        args=_get_args(app_env, module_name, framework_variable_name, port),
        resources=get_proto_resources(app_env.resources),
        ports=container_ports,
        env=[literals_pb2.KeyValuePair(key=k, value=v) for k, v in _get_env(app_env).items()],
    )


def _sanitize_resource_name(resource: tasks_pb2.Resources.ResourceEntry) -> str:
    """
    Sanitize resource name for Kubernetes compatibility.

    Args:
        resource: Resource entry

    Returns:
        Sanitized resource name
    """
    return tasks_pb2.Resources.ResourceName.Name(resource.name).lower().replace("_", "-")


def _serialized_pod_spec(
    app_env: AppEnvironment,
    pod_template: PodTemplate,
    serialization_context: SerializationContext,
    additional_distribution: Optional[str],
    port: int,
    module_name: Optional[str],
    framework_variable_name: Optional[str],
) -> dict:
    """
    Convert pod spec into a dict for serialization.

    Args:
        app_env: The app environment
        pod_template: Pod template specification
        serialization_context: Serialization context
        additional_distribution: URL to additional distribution files
        port: Port number
        module_name: Name of the module containing the app
        framework_variable_name: Variable name of the framework app

    Returns:
        Dictionary representation of the pod spec
    """
    from kubernetes.client import ApiClient
    from kubernetes.client.models import V1Container, V1ContainerPort, V1EnvVar, V1ResourceRequirements

    pod_template = deepcopy(pod_template)

    if pod_template.pod_spec is None:
        return {}

    if pod_template.primary_container_name != "app":
        msg = "Primary container name must be 'app'"
        raise ValueError(msg)

    containers: list[V1Container] = pod_template.pod_spec.containers
    primary_exists = any(container.name == pod_template.primary_container_name for container in containers)

    if not primary_exists:
        msg = "Primary container does not exist with name 'app'"
        raise ValueError(msg)

    final_containers = []

    # Process containers
    for container in containers:
        container.image = _get_image_uri(container.image, serialization_context)

        if container.name == pod_template.primary_container_name:
            container.args = _get_args(app_env, module_name, framework_variable_name, port)
            container.command = _get_command(app_env, additional_distribution)

            limits, requests = {}, {}
            resources = get_proto_resources(app_env.resources)
            if resources:
                for resource in resources.limits:
                    limits[_sanitize_resource_name(resource)] = resource.value
                for resource in resources.requests:
                    requests[_sanitize_resource_name(resource)] = resource.value

                resource_requirements = V1ResourceRequirements(limits=limits, requests=requests)

                if limits or requests:
                    container.resources = resource_requirements

            if app_env.env_vars:
                container.env = [V1EnvVar(name=k, value=v) for k, v in app_env.env_vars.items()] + (
                    container.env or []
                )

            from flyte.app._types import Port
            _port = Port(port=port, name="http")
            container.ports = [V1ContainerPort(container_port=_port.port, name=_port.name)] + (
                container.ports or []
            )

        final_containers.append(container)

    pod_template.pod_spec.containers = final_containers
    return ApiClient().sanitize_for_serialization(pod_template.pod_spec)


def _get_k8s_pod(
    app_env: AppEnvironment,
    pod_template: PodTemplate,
    serialization_context: SerializationContext,
    additional_distribution: Optional[str],
    port: int,
    module_name: Optional[str],
    framework_variable_name: Optional[str],
) -> tasks_pb2.K8sPod:
    """
    Convert pod_template into a K8sPod IDL.

    Args:
        app_env: The app environment
        pod_template: Pod template specification
        serialization_context: Serialization context
        additional_distribution: URL to additional distribution files
        port: Port number
        module_name: Name of the module containing the app
        framework_variable_name: Variable name of the framework app

    Returns:
        K8sPod protobuf message
    """
    import json

    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Struct

    pod_spec_dict = _serialized_pod_spec(
        app_env,
        pod_template,
        serialization_context,
        additional_distribution,
        port,
        module_name,
        framework_variable_name
    )
    pod_spec_idl = Parse(json.dumps(pod_spec_dict), Struct())

    metadata = tasks_pb2.K8sObjectMetadata(
        labels=pod_template.labels,
        annotations=pod_template.annotations,
    )
    return tasks_pb2.K8sPod(pod_spec=pod_spec_idl, metadata=metadata)


def _get_scaling_metric(metric: Any) -> Optional[app_definition_pb2.ScalingMetric]:
    """
    Convert scaling metric to protobuf format.

    Args:
        metric: Scaling metric (Concurrency or RequestRate)

    Returns:
        ScalingMetric protobuf message or None
    """
    from flyte.app._types import Scaling

    if metric is None:
        return None

    if isinstance(metric, Scaling.Concurrency):
        return app_definition_pb2.ScalingMetric(
            concurrency=app_definition_pb2.Concurrency(val=metric.val)
        )
    elif isinstance(metric, Scaling.RequestRate):
        return app_definition_pb2.ScalingMetric(
            request_rate=app_definition_pb2.RequestRate(val=metric.val)
        )

    return None


def translate_app_env_to_idl(
    app_env: AppEnvironment,
    serialization_context: SerializationContext,
    additional_distribution: Optional[str] = None,
    desired_state: app_definition_pb2.Spec.DesiredState.ValueType = app_definition_pb2.Spec.DesiredState.DESIRED_STATE_ACTIVE,
    port: int = 8080,
    module_name: Optional[str] = None,
    framework_variable_name: Optional[str] = None,
) -> app_definition_pb2.App:
    """
    Translate an AppEnvironment to AppIDL protobuf format.

    This is the main entry point for serializing an AppEnvironment object into
    the AppIDL protobuf format.

    Args:
        app_env: The app environment to serialize
        serialization_context: Serialization context containing org, project, domain, version, etc.
        additional_distribution: URL to additional distribution files (e.g., tar.gz of include files)
        desired_state: Desired state of the app (ACTIVE, INACTIVE, etc.)
        port: Port number for the app (default: 8080)
        module_name: Name of the module containing the app (for framework apps)
        framework_variable_name: Variable name of the framework app (for framework apps)

    Returns:
        AppIDL protobuf message
    """
    # Build security context
    security_context_kwargs = {}
    security_context = None
    if app_env.secrets:
        security_context_kwargs["secrets"] = [s.to_flyte_idl() for s in app_env.secrets]
    if not app_env.requires_auth:
        security_context_kwargs["allow_anonymous"] = True

    if security_context_kwargs:
        security_context = app_definition_pb2.SecurityContext(**security_context_kwargs)

    # Build autoscaling config
    scaling_metric = _get_scaling_metric(app_env.scaling.metric)

    dur = None
    if app_env.scaling.scaledown_after:
        dur = Duration()
        dur.FromTimedelta(app_env.scaling.scaledown_after)

    min_replicas, max_replicas = app_env.scaling.replicas
    autoscaling = app_definition_pb2.AutoscalingConfig(
        replicas=app_definition_pb2.Replicas(min=min_replicas, max=max_replicas),
        scaledown_period=dur,
        scaling_metric=scaling_metric,
    )

    # Build spec based on image type
    container = None
    pod = None
    # TODO check pod template
    if isinstance(app_env.image, (str, Image)):
        container = _get_container(
            app_env,
            serialization_context,
            additional_distribution,
            port,
            module_name,
            framework_variable_name,
        )
    elif isinstance(app_env.image, PodTemplate):
         pod = _get_k8s_pod(
            app_env,
            app_env.image,
            serialization_context,
            additional_distribution,
            port,
            module_name,
            framework_variable_name,
        )
    else:
        msg = "image must be a str, Image, or PodTemplate"
        raise ValueError(msg)

    # Build ingress config
    subdomain = app_env.domain.subdomain if app_env.domain else None
    custom_domain = app_env.domain.custom_domain if app_env.domain else None

    ingress = app_definition_pb2.IngressConfig(
        private=False,
        subdomain=subdomain if subdomain else None,
        cname=custom_domain if custom_domain else None,
    )

    # Build links
    links = None
    if app_env.links:
        links = [
            app_definition_pb2.Link(path=link.path, title=link.title, is_relative=link.is_relative)
            for link in app_env.links
        ]

    # Build profile
    profile = app_definition_pb2.Profile(
        type=app_env.type,
        short_description=app_env.docs,
    )

    # Build the full App IDL
    return app_definition_pb2.App(
        metadata=app_definition_pb2.Meta(
            id=app_definition_pb2.Identifier(
                org=serialization_context.org,
                project=serialization_context.project,
                domain=serialization_context.domain,
                name=app_env.name,
            ),
        ),
        spec=app_definition_pb2.Spec(
            desired_state=desired_state,
            ingress=ingress,
            autoscaling=autoscaling,
            security_context=security_context,
            cluster_pool=app_env.cluster_pool,
            extended_resources=get_proto_extended_resources(app_env.resources),
            runtime_metadata=runtime_version_pb2.RuntimeMetadata(
                type=runtime_version_pb2.RuntimeMetadata.RuntimeType.FLYTE_SDK,
                version=serialization_context.version,
                flavor="python",
            ),
            profile=profile,
            links=links,
            container=container,
            pod=pod
        ),
    )
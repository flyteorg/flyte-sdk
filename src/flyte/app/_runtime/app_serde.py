"""
Serialization module for AppEnvironment to AppIDL conversion.

This module provides functionality to serialize an AppEnvironment object into
the AppIDL protobuf format, using SerializationContext for configuration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Union

from flyteidl2.app import app_definition_pb2
from flyteidl2.common import runtime_version_pb2
from flyteidl2.core import literals_pb2, tasks_pb2
from google.protobuf.duration_pb2 import Duration

import flyte
import flyte.errors
import flyte.io
from flyte._internal.runtime.resources_serde import get_proto_extended_resources, get_proto_resources
from flyte._internal.runtime.task_serde import get_security_context, lookup_image_in_cache
from flyte.app import AppEnvironment, Input, Scaling
from flyte.models import SerializationContext


def get_proto_container(
    app_env: AppEnvironment,
    serialization_context: SerializationContext,
) -> tasks_pb2.Container:
    """
    Construct the container specification.

    Args:
        app_env: The app environment
        serialization_context: Serialization context

    Returns:
        Container protobuf message
    """
    from flyte import Image

    env = [literals_pb2.KeyValuePair(key=k, value=v) for k, v in app_env.env_vars.items()] if app_env.env_vars else None
    resources = get_proto_resources(app_env.resources)

    if app_env.image == "auto":
        img = Image.from_debian_base()
    elif isinstance(app_env.image, str):
        img = Image.from_base(app_env.image)
    else:
        img = app_env.image

    env_name = app_env.name
    img_uri = lookup_image_in_cache(serialization_context, env_name, img)

    p = app_env.get_port()
    container_ports = [tasks_pb2.ContainerPort(container_port=p.port, name=p.name)]

    return tasks_pb2.Container(
        image=img_uri,
        command=app_env.container_cmd(serialization_context),
        args=app_env.container_args(serialization_context),
        resources=resources,
        ports=container_ports,
        env=env,
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
    pod_template: flyte.PodTemplate,
    serialization_context: SerializationContext,
) -> dict:
    """
    Convert pod spec into a dict for serialization.

    Args:
        app_env: The app environment
        pod_template: Pod template specification
        serialization_context: Serialization context

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
        img = container.image
        if isinstance(img, flyte.Image):
            img = lookup_image_in_cache(serialization_context, container.name, img)
        container.image = img

        if container.name == pod_template.primary_container_name:
            container.args = app_env.container_args(serialization_context)
            container.command = app_env.container_cmd(serialization_context)

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
                container.env = [V1EnvVar(name=k, value=v) for k, v in app_env.env_vars.items()] + (container.env or [])

            _port = app_env.get_port()
            container.ports = [V1ContainerPort(container_port=_port.port, name=_port.name)] + (container.ports or [])

        final_containers.append(container)

    pod_template.pod_spec.containers = final_containers
    return ApiClient().sanitize_for_serialization(pod_template.pod_spec)


def _get_k8s_pod(
    app_env: AppEnvironment,
    pod_template: flyte.PodTemplate,
    serialization_context: SerializationContext,
) -> tasks_pb2.K8sPod:
    """
    Convert pod_template into a K8sPod IDL.

    Args:
        app_env: The app environment
        pod_template: Pod template specification
        serialization_context: Serialization context

    Returns:
        K8sPod protobuf message
    """
    import json

    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Struct

    pod_spec_dict = _serialized_pod_spec(app_env, pod_template, serialization_context)
    pod_spec_idl = Parse(json.dumps(pod_spec_dict), Struct())

    metadata = tasks_pb2.K8sObjectMetadata(
        labels=pod_template.labels,
        annotations=pod_template.annotations,
    )
    return tasks_pb2.K8sPod(pod_spec=pod_spec_idl, metadata=metadata)


def _get_scaling_metric(
    metric: Optional[Union[Scaling.Concurrency, Scaling.RequestRate]],
) -> Optional[app_definition_pb2.ScalingMetric]:
    """
    Convert scaling metric to protobuf format.

    Args:
        metric: Scaling metric (Concurrency or RequestRate)

    Returns:
        ScalingMetric protobuf message or None
    """

    if metric is None:
        return None

    if isinstance(metric, Scaling.Concurrency):
        return app_definition_pb2.ScalingMetric(concurrency=app_definition_pb2.Concurrency(val=metric.val))
    elif isinstance(metric, Scaling.RequestRate):
        return app_definition_pb2.ScalingMetric(request_rate=app_definition_pb2.RequestRate(val=metric.val))

    return None


def translate_inputs(inputs: List[Input]) -> app_definition_pb2.InputList:
    """
    Placeholder for translating inputs to protobuf format.

    Returns:
        InputList protobuf message
    """
    if not inputs:
        return app_definition_pb2.InputList()

    inputs_list = []
    for input in inputs:
        if isinstance(input.value, str):
            inputs_list.append(app_definition_pb2.Input(name=input.name, string_value=input.value))
        elif isinstance(input.value, flyte.io.File):
            inputs_list.append(app_definition_pb2.Input(name=input.name, string_value=str(input.value.path)))
        elif isinstance(input.value, flyte.io.Dir):
            inputs_list.append(app_definition_pb2.Input(name=input.name, string_value=str(input.value.path)))
        else:
            raise ValueError(f"Unsupported input value type: {type(input.value)}")
    return app_definition_pb2.InputList(items=inputs_list)


def translate_app_env_to_idl(
    app_env: AppEnvironment,
    serialization_context: SerializationContext,
    desired_state: app_definition_pb2.Spec.DesiredState = app_definition_pb2.Spec.DesiredState.DESIRED_STATE_ACTIVE,
) -> app_definition_pb2.App:
    """
    Translate an AppEnvironment to AppIDL protobuf format.

    This is the main entry point for serializing an AppEnvironment object into
    the AppIDL protobuf format.

    Args:
        app_env: The app environment to serialize
        serialization_context: Serialization context containing org, project, domain, version, etc.
        desired_state: Desired state of the app (ACTIVE, INACTIVE, etc.)

    Returns:
        AppIDL protobuf message
    """
    # Build security context
    task_sec_ctx = get_security_context(app_env.secrets)
    allow_anonymous = False
    if not app_env.requires_auth:
        allow_anonymous = True

    security_context = None
    if task_sec_ctx or allow_anonymous:
        security_context = app_definition_pb2.SecurityContext(
            run_as=task_sec_ctx.run_as if task_sec_ctx else None,
            secrets=task_sec_ctx.secrets if task_sec_ctx else [],
            allow_anonymous=allow_anonymous,
        )

    # Build autoscaling config
    scaling_metric = _get_scaling_metric(app_env.scaling.metric)

    dur = None
    if app_env.scaling.scaledown_after:
        dur = Duration()
        dur.FromTimedelta(app_env.scaling.scaledown_after)

    min_replicas, max_replicas = app_env.scaling.get_replicas()
    autoscaling = app_definition_pb2.AutoscalingConfig(
        replicas=app_definition_pb2.Replicas(min=min_replicas, max=max_replicas),
        scaledown_period=dur,
        scaling_metric=scaling_metric,
    )

    # Build spec based on image type
    container = None
    pod = None
    if app_env.pod_template:
        if isinstance(app_env.pod_template, str):
            raise NotImplementedError("PodTemplate as str is not supported yet")
        pod = _get_k8s_pod(
            app_env,
            app_env.pod_template,
            serialization_context,
        )
    elif app_env.image:
        container = get_proto_container(
            app_env,
            serialization_context,
        )
    else:
        msg = "image must be a str, Image, or PodTemplate"
        raise ValueError(msg)

    ingress = app_definition_pb2.IngressConfig(
        private=False,
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
        short_description=app_env.description,
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
                version=flyte.version(),
                flavor="python",
            ),
            profile=profile,
            links=links,
            container=container,
            pod=pod,
            inputs=translate_inputs(app_env.inputs),
        ),
    )

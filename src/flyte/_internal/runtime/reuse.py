import hashlib
import typing
from venv import logger

from flyteidl2.core import tasks_pb2

import flyte.errors
from flyte import ReusePolicy
from flyte._pod import _PRIMARY_CONTAINER_DEFAULT_NAME, _PRIMARY_CONTAINER_NAME_FIELD
from flyte.models import CodeBundle


def extract_unique_id_and_image(
    env_name: str,
    code_bundle: CodeBundle | None,
    task: tasks_pb2.TaskTemplate,
    reuse_policy: ReusePolicy,
) -> typing.Tuple[str, str]:
    """
    Compute a unique ID for the task based on its name, version, image URI, and code bundle.
    :param env_name: Name of the reusable environment.
    :param reuse_policy: The reuse policy for the task.
    :param task: The task template.
    :param code_bundle: The code bundle associated with the task.
    :return: A unique ID string and the image URI.
    """
    image = ""
    container_ser = ""
    if task.HasField("container"):
        copied_container = tasks_pb2.Container()
        copied_container.CopyFrom(task.container)
        copied_container.args.clear()  # Clear args to ensure deterministic serialization
        container_ser = copied_container.SerializeToString(deterministic=True)
        image = copied_container.image

    if task.HasField("k8s_pod"):
        # Clear args to ensure deterministic serialization
        copied_k8s_pod = tasks_pb2.K8sPod()
        copied_k8s_pod.CopyFrom(task.k8s_pod)
        if task.config is not None:
            primary_container_name = task.config[_PRIMARY_CONTAINER_NAME_FIELD]
        else:
            primary_container_name = _PRIMARY_CONTAINER_DEFAULT_NAME
        for container in copied_k8s_pod.pod_spec["containers"]:
            if "name" in container and container["name"] == primary_container_name:
                image = container["image"]
                del container["args"]
        container_ser = copied_k8s_pod.SerializeToString(deterministic=True)

    components = f"{env_name}:{container_ser}"
    if isinstance(reuse_policy.replicas, tuple):
        components += f":{reuse_policy.replicas[0]}:{reuse_policy.replicas[1]}"
    else:
        components += f":{reuse_policy.replicas}"
    if reuse_policy.ttl is not None:
        components += f":{reuse_policy.ttl.total_seconds()}"
    if reuse_policy.get_scaledown_ttl() is not None:
        components += f":{reuse_policy.get_scaledown_ttl()}"
    if code_bundle is not None:
        components += f":{code_bundle.computed_version}"
    if task.security_context is not None:
        security_ctx_str = task.security_context.SerializeToString(deterministic=True)
        components += f":{security_ctx_str}"
    if task.metadata.interruptible is not None:
        components += f":{task.metadata.interruptible}"
    if task.metadata.pod_template_name is not None:
        components += f":{task.metadata.pod_template_name}"
    sha256 = hashlib.sha256()
    sha256.update(components.encode("utf-8"))
    return sha256.hexdigest(), image


def add_reusable(
    task: tasks_pb2.TaskTemplate,
    reuse_policy: ReusePolicy,
    code_bundle: CodeBundle | None,
    parent_env_name: str | None = None,
) -> tasks_pb2.TaskTemplate:
    """
    Convert a ReusePolicy to a custom configuration dictionary.

    :param task: The task to which the reusable policy will be added.
    :param reuse_policy: The reuse policy to apply.
    :param code_bundle: The code bundle associated with the task.
    :param parent_env_name: The name of the parent environment, if any.
    :return: The modified task with the reusable policy added.
    """
    if reuse_policy is None:
        return task

    if task.HasField("custom"):
        raise flyte.errors.RuntimeUserError(
            "BadConfiguration", "Plugins do not support reusable policy. Only container tasks and pods."
        )

    logger.debug(f"Adding reusable policy for task: {task.id.name}")
    name = parent_env_name if parent_env_name else ""
    if parent_env_name is None:
        name = task.id.name.split(".")[0]

    version, image_uri = extract_unique_id_and_image(
        env_name=name, code_bundle=code_bundle, task=task, reuse_policy=reuse_policy
    )

    scaledown_ttl = reuse_policy.get_scaledown_ttl()

    task.custom = {
        "name": name,
        "version": version[:15],  # Use only the first 15 characters for the version
        "type": "actor",
        "spec": {
            "container_image": image_uri,
            "backlog_length": None,
            "parallelism": reuse_policy.concurrency,
            "min_replica_count": reuse_policy.min_replicas,
            "replica_count": reuse_policy.max_replicas,
            "ttl_seconds": reuse_policy.ttl.total_seconds() if reuse_policy.ttl else None,
            "scaledown_ttl_seconds": scaledown_ttl.total_seconds() if scaledown_ttl else None,
        },
    }

    task.type = "actor"
    logger.info(f"Reusable task {task.id.name} with config {task.custom}")

    return task

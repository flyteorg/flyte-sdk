import datetime
import random

import pytest
from kubernetes.client import ApiClient, V1Container, V1EnvVar, V1LocalObjectReference, V1PodSpec

from flyte import ReusePolicy
from flyte._internal.runtime.reuse import add_reusable, extract_unique_id_and_image
from flyte._pod import _PRIMARY_CONTAINER_NAME_FIELD
from flyte.models import CodeBundle
from flyteidl2.core import security_pb2, tasks_pb2


@pytest.fixture
def code_bundle():
    """Creates a CodeBundle with a known computed_version."""
    return CodeBundle(
        computed_version="test123",
        tgz="test-bundle.tgz",
    )


@pytest.fixture
def container_task():
    """Creates a TaskTemplate with a container."""
    task_template = tasks_pb2.TaskTemplate()
    task_template.id.name = "test-task"
    task_template.id.version = "v1"
    task_template.type = "python-task"

    # Add container
    container = tasks_pb2.Container()
    container.image = "test-image:latest"
    container.args.extend(["arg1", "arg2"])  # These should be cleared in extract_unique_id_and_image
    task_template.container.CopyFrom(container)

    return task_template


@pytest.fixture
def k8s_pod_task():
    """Creates a TaskTemplate with a K8sPod."""
    task_template = tasks_pb2.TaskTemplate()
    task_template.id.name = "test-pod-task"
    task_template.id.version = "v1"
    task_template.type = "python-task"

    pod_spec = V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                env=[V1EnvVar(name="hello", value="world")],
                image="pod-image:latest",
                args=["arg1", str(random.randint)],
            )
        ],
        image_pull_secrets=[V1LocalObjectReference(name="regcred-test")],
    )
    # Add K8sPod
    k8s_pod = tasks_pb2.K8sPod(
        pod_spec=ApiClient().sanitize_for_serialization(pod_spec),
    )
    task_template.k8s_pod.CopyFrom(k8s_pod)

    # Set primary container name
    task_template.config.update({_PRIMARY_CONTAINER_NAME_FIELD: "primary"})

    return task_template


@pytest.fixture
def reuse_policy():
    """Creates a ReusePolicy."""
    return ReusePolicy(
        replicas=(1, 3),  # min, max
        idle_ttl=datetime.timedelta(minutes=30),
    )


@pytest.fixture
def reuse_policy_scaledown():
    """Creates a ReusePolicy."""
    return ReusePolicy(
        replicas=(1, 3),  # min, max
        idle_ttl=datetime.timedelta(seconds=90),  # 90 seconds for idle TTL
        scaledown_ttl=datetime.timedelta(seconds=30),  # 30 seconds for scaledown TTL
        concurrency=10,  # Example concurrency setting
    )


def test_extract_unique_id_container(container_task, code_bundle, reuse_policy):
    """Test extracting unique ID from a task with container."""
    unique_id, image = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=container_task,
        reuse_policy=reuse_policy,
    )

    assert isinstance(unique_id, str)
    assert len(unique_id) == 64  # SHA-256 hexdigest length
    assert image == "test-image:latest"


def test_extract_unique_id_k8s_pod(k8s_pod_task, code_bundle, reuse_policy):
    """Test extracting unique ID from a task with K8sPod."""
    unique_id, image = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=k8s_pod_task,
        reuse_policy=reuse_policy,
    )

    assert isinstance(unique_id, str)
    assert len(unique_id) == 64  # SHA-256 hexdigest length
    assert image == "pod-image:latest"


def test_extract_unique_id_with_security_context(container_task, code_bundle, reuse_policy):
    """Test that security context affects the unique ID."""
    # First get a baseline ID
    baseline_id, _ = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=container_task,
        reuse_policy=reuse_policy,
    )

    # Now add a security context and extract again
    security_context = security_pb2.SecurityContext(
        secrets=[
            security_pb2.Secret(
                group="test-group",
            ),
            security_pb2.Secret(
                group="another-group",
            ),
        ],
    )
    container_task.security_context.CopyFrom(security_context)

    new_id, _ = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=container_task,
        reuse_policy=reuse_policy,
    )

    # The IDs should be different because the security context was added
    assert baseline_id != new_id


def test_extract_unique_id_with_interruptible(container_task, code_bundle, reuse_policy):
    """Test that interruptible flag affects the unique ID."""
    # First get a baseline ID
    baseline_id, _ = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=container_task,
        reuse_policy=reuse_policy,
    )

    # Now set interruptible to True and extract again
    container_task.metadata.interruptible = True

    new_id, _ = extract_unique_id_and_image(
        env_name="test-env",
        code_bundle=code_bundle,
        task=container_task,
        reuse_policy=reuse_policy,
    )

    # The IDs should be different because interruptible was enabled
    assert baseline_id != new_id


def test_add_reusable_container_task(container_task, code_bundle, reuse_policy):
    """Test adding reusable policy to a container task."""
    modified_task = add_reusable(
        task=container_task,
        reuse_policy=reuse_policy,
        code_bundle=code_bundle,
    )

    # Check that custom field was added correctly
    assert modified_task.custom is not None
    assert modified_task.custom["name"] == "test-task"
    assert "version" in modified_task.custom
    assert modified_task.custom["type"] == "actor"
    assert modified_task.custom["spec"]["container_image"] == "test-image:latest"
    assert modified_task.custom["spec"]["replica_count"] == 3
    assert modified_task.custom["spec"]["ttl_seconds"] == 1800  # 30 minutes in seconds
    assert modified_task.custom["spec"]["min_replica_count"] == 1
    assert modified_task.custom["spec"]["parallelism"] == 1
    assert modified_task.custom["spec"]["scaledown_ttl_seconds"] == 30  # No scaledown TTL set


def test_add_reusable_with_parent_env(container_task, code_bundle, reuse_policy):
    """Test adding reusable policy with a parent environment name."""
    modified_task = add_reusable(
        task=container_task,
        reuse_policy=reuse_policy,
        code_bundle=code_bundle,
        parent_env_name="parent-env",
    )

    # Check that custom field uses the parent env name
    assert modified_task.custom["name"] == "parent-env"


def test_add_reusable_with_existing_custom_raises_error(container_task, code_bundle, reuse_policy):
    """Test that adding reusable policy to a task with existing custom config raises error."""
    # Add custom field to the task
    container_task.custom = {"existing": "config"}

    # Should raise an error
    with pytest.raises(Exception) as excinfo:
        add_reusable(
            task=container_task,
            reuse_policy=reuse_policy,
            code_bundle=code_bundle,
        )

    assert "Plugins do not support reusable policy. Only container tasks and pods." in str(excinfo.value)


def test_add_reusable_none_policy(container_task, code_bundle):
    """Test that passing None as reuse policy returns the task unchanged."""
    original_task = tasks_pb2.TaskTemplate()
    original_task.CopyFrom(container_task)

    modified_task = add_reusable(
        task=container_task,
        reuse_policy=None,
        code_bundle=code_bundle,
    )

    # Task should be unchanged
    assert modified_task == original_task


def test_add_reusable_scaledown_ttl(reuse_policy_scaledown):
    """Test that scaledown_ttl is correctly set in the reusable task."""
    task_template = tasks_pb2.TaskTemplate()
    task_template.id.name = "test-task"
    task_template.id.version = "v1"

    modified_task = add_reusable(
        task=task_template,
        reuse_policy=reuse_policy_scaledown,
        code_bundle=None,
    )

    assert modified_task.custom["spec"]["scaledown_ttl_seconds"] == reuse_policy_scaledown.scaledown_ttl.total_seconds()
    assert modified_task.custom["spec"]["parallelism"] == reuse_policy_scaledown.concurrency

import pathlib

import pytest
from flyteidl2.task import task_definition_pb2
from kubernetes.client import V1Container, V1EnvVar, V1PodSpec

import flyte
from flyte import PodTemplate, RetryStrategy
from flyte._internal.runtime.task_serde import get_proto_task, get_security_context, translate_task_to_wire
from flyte.models import SerializationContext
from flyte.remote._task import TaskDetails

env = flyte.TaskEnvironment(name="hello_world", resources=flyte.Resources(cpu=1, memory="250Mi"))


@env.task
async def oomer(x: int) -> int:
    pass


env_with_reuse = flyte.TaskEnvironment(
    name="oomer_with_reuse",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
)


@env_with_reuse.task
async def oomer_with_reuse(x: int) -> int:
    pass


def test_oomer_override():
    """
    Test the override functionality of the oomer task.
    """
    pod_template = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary", env=[V1EnvVar(name="hello", value="world")])],
        ),
    )
    # Create a new task with overridden resources
    new_task = oomer.override(
        resources=flyte.Resources(cpu=2, memory="500Mi"), pod_template=pod_template, short_name="new_oomer"
    )

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 2
    assert new_task.resources.memory == "500Mi"
    assert new_task.pod_template == pod_template
    assert new_task.short_name == "new_oomer"
    assert isinstance(new_task.cache, flyte.Cache)

    # Check if the new task is not the same as the original task
    assert new_task != oomer


def test_oomer_override_with_reuse_incorrect():
    """
    Test the override functionality of the oomer task with reuse.
    """
    # Create a new task with overridden resources and reuse policy
    with pytest.raises(ValueError):
        oomer.override(
            resources=flyte.Resources(cpu=2, memory="500Mi"),
            reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            resources=flyte.Resources(cpu=2, memory="500Mi"),
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            env_vars={},
        )

    with pytest.raises(ValueError):
        oomer_with_reuse.override(
            secrets="my_secret",
        )


def test_override_with_reuse():
    """
    Test the override functionality of the oomer task with reuse.
    """
    # Create a new task with overridden resources and reuse policy
    new_task = oomer_with_reuse.override(
        cache=flyte.Cache("auto"),
    )

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 1
    assert new_task.resources.memory == "250Mi"
    assert isinstance(new_task.cache, flyte.Cache)

    # Check if the new task is not the same as the original task
    assert new_task != oomer_with_reuse


def test_override_turn_reuse_off():
    """
    Test the override functionality of the oomer task with reuse turned off.
    """
    # Create a new task with reuse turned off
    new_task = oomer_with_reuse.override(reusable="off", resources=flyte.Resources(cpu=2, memory="500Mi"))

    # Check if the new task has the correct resources
    assert new_task.resources.cpu == 2
    assert new_task.resources.memory == "500Mi"
    assert new_task.reusable is None

    # Check if the new task is not the same as the original task
    assert new_task != oomer_with_reuse


pod_template_with_labels = PodTemplate(
    pod_spec=V1PodSpec(
        containers=[V1Container(name="primary", env=[V1EnvVar(name="hello", value="world")])],
    ),
    labels={"team": "ml"},
    annotations={"note": "testing"},
)


env_with_pod_template = flyte.TaskEnvironment(
    name="env_with_pod",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    pod_template=pod_template_with_labels,
)


@env_with_pod_template.task
async def task_with_env_pod(x: int) -> int:
    pass


env_reuse_with_pod = flyte.TaskEnvironment(
    name="env_reuse_pod",
    resources=flyte.Resources(cpu=1, memory="250Mi"),
    reusable=flyte.ReusePolicy(replicas=2, idle_ttl=60),
)


@env_reuse_with_pod.task
async def task_reuse_with_pod(x: int) -> int:
    pass


def test_override_short_name_preserves_env_pod_template():
    """
    When a task gets its pod_template from the environment and only
    short_name is overridden, the pod_template should be preserved.
    """
    assert task_with_env_pod.pod_template == pod_template_with_labels

    new_task = task_with_env_pod.override(short_name="renamed_task")

    assert new_task.short_name == "renamed_task"
    assert new_task.pod_template == pod_template_with_labels
    assert new_task.pod_template is not None
    assert new_task.pod_template.labels == {"team": "ml"}
    assert new_task.pod_template.annotations == {"note": "testing"}
    # Original task should be unchanged
    assert task_with_env_pod.short_name == "task_with_env_pod"


def test_override_short_name_preserves_inline_pod_template():
    """
    When a task has a pod_template set via override() and then short_name
    is overridden again, the pod_template should be preserved.
    """
    pod = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
        ),
        labels={"stage": "prod"},
    )
    task_with_pod = oomer.override(pod_template=pod, short_name="first_override")
    assert task_with_pod.pod_template == pod

    # Now override only short_name — pod_template must survive
    renamed = task_with_pod.override(short_name="second_override")
    assert renamed.short_name == "second_override"
    assert renamed.pod_template == pod
    assert renamed.pod_template.labels == {"stage": "prod"}


def test_reuse_policy_with_pod_template_override_short_name():
    """
    A task with reuse policy should preserve the pod_template when
    short_name is overridden (and no disallowed fields are changed).
    """
    # Give the task a pod_template by turning off reuse first, then re-enabling
    # Or: override only short_name on a reusable task (allowed)
    new_task = task_reuse_with_pod.override(short_name="renamed_reuse")

    assert new_task.short_name == "renamed_reuse"
    assert new_task.reusable is not None
    assert new_task.reusable.replicas == (2, 2)


def test_reuse_off_with_pod_template_and_short_name():
    """
    When reuse is turned off and both pod_template and short_name are set,
    everything should be preserved together.
    """
    pod = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary")],
        ),
        labels={"env": "staging"},
    )
    new_task = task_reuse_with_pod.override(
        reusable="off",
        resources=flyte.Resources(cpu=4, memory="1Gi"),
        pod_template=pod,
        short_name="no_reuse_task",
    )

    assert new_task.short_name == "no_reuse_task"
    assert new_task.reusable is None
    assert new_task.resources.cpu == 4
    assert new_task.pod_template == pod
    assert new_task.pod_template.labels == {"env": "staging"}


def test_serialize_task_with_env_pod_template_and_short_name():
    """
    Serializing a task that has pod_template from the environment and an
    overridden short_name should preserve both in the wire format.
    """
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    new_task = task_with_env_pod.override(short_name="serialized_task")
    assert new_task.pod_template is not None

    task_spec = translate_task_to_wire(new_task, context)

    # short_name should be preserved in TaskSpec
    assert task_spec.short_name == "serialized_task"

    # pod_template should result in a k8s_pod (not a container)
    task_template = task_spec.task_template
    assert task_template.k8s_pod is not None
    assert task_template.container.image == ""  # container should be empty when pod is set

    # Labels and annotations should be preserved
    assert task_template.k8s_pod.metadata.labels == {"team": "ml"}
    assert task_template.k8s_pod.metadata.annotations == {"note": "testing"}

    # primary_container_name should be in config
    assert task_template.config["primary_container_name"] == "primary"


def test_serialize_task_reuse_off_pod_template_short_name():
    """
    Serializing a task where reuse was turned off and pod_template + short_name
    were overridden should preserve all fields correctly.
    """
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    pod = PodTemplate(
        pod_spec=V1PodSpec(
            containers=[V1Container(name="primary", env=[V1EnvVar(name="MY_VAR", value="123")])],
        ),
        labels={"app": "worker"},
        annotations={"version": "v2"},
    )
    new_task = task_reuse_with_pod.override(
        reusable="off",
        resources=flyte.Resources(cpu=2, memory="512Mi"),
        pod_template=pod,
        short_name="serde_reuse_off",
    )

    task_spec = translate_task_to_wire(new_task, context)

    assert task_spec.short_name == "serde_reuse_off"

    task_template = task_spec.task_template
    assert task_template.k8s_pod is not None
    assert task_template.k8s_pod.metadata.labels == {"app": "worker"}
    assert task_template.k8s_pod.metadata.annotations == {"version": "v2"}
    assert task_template.config["primary_container_name"] == "primary"

    # Reusable should not be set (no reuse plugin config)
    # The task should NOT have been processed by add_reusable
    assert new_task.reusable is None


def test_serialize_preserves_pod_template_after_multiple_overrides():
    """
    After multiple chained overrides, serialization should still produce
    a correct k8s_pod with the pod_template from the environment.
    """
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    # First override: change short_name
    t1 = task_with_env_pod.override(short_name="step_one")
    assert t1.pod_template == pod_template_with_labels

    # Second override: change short_name again
    t2 = t1.override(short_name="step_two")
    assert t2.pod_template == pod_template_with_labels

    task_spec = translate_task_to_wire(t2, context)
    assert task_spec.short_name == "step_two"
    assert task_spec.task_template.k8s_pod is not None
    assert task_spec.task_template.k8s_pod.metadata.labels == {"team": "ml"}


def test_override_ref_task():
    context = SerializationContext(
        project="test-project",
        domain="test-domain",
        version="test-version",
        org="test-org",
        input_path="/tmp/inputs",
        output_path="/tmp/outputs",
        image_cache=None,
        code_bundle=None,
        root_dir=pathlib.Path.cwd(),
    )

    # Generate proto task
    new_task = oomer_with_reuse.override(reusable="off", resources=flyte.Resources(cpu=2, memory="500Mi"))
    task_template = get_proto_task(new_task, context)

    task_details_pb2 = task_definition_pb2.TaskDetails(spec=task_definition_pb2.TaskSpec(task_template=task_template))
    td = TaskDetails(pb2=task_details_pb2)

    secrets = [flyte.Secret(key="openai", as_env_var="OPENAI_API_KEY")]
    new_td = td.override(
        short_name="new_oomer",
        resources=flyte.Resources(cpu=3, memory="100Mi"),
        retries=RetryStrategy(5),
        timeout=100,
        env_vars={"FOO": "BAR"},
        secrets=secrets,
    )
    assert new_td is not td
    assert new_td is not None
    assert new_td.resources[0][0].value == "3"
    assert new_td.resources[0][1].value == "100Mi"
    assert new_td.pb2.spec.short_name == "new_oomer"
    assert new_td.pb2.spec.task_template.metadata.retries.retries == 5
    assert new_td.pb2.spec.task_template.metadata.timeout.seconds == 100
    assert new_td.pb2.spec.task_template.security_context == get_security_context(secrets)

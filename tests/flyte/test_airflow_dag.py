"""
Tests for the Airflow DAG monkey-patch in flyteplugins.airflow.dag.
"""
import pytest

import flyte
from flyte._image import Image


@pytest.fixture(autouse=True)
def _reset_environment_registry():
    """Clean up TaskEnvironment instances created during each test."""
    from flyte._environment import _ENVIRONMENT_REGISTRY
    initial_len = len(_ENVIRONMENT_REGISTRY)
    yield
    del _ENVIRONMENT_REGISTRY[initial_len:]


def test_flyte_env_image_preserved_after_dag_build():
    """The image supplied via flyte_env= must survive FlyteDAG.build().

    Previously, build() called TaskEnvironment.from_task() which derived a new
    environment from the operator tasks' images (all None), silently discarding
    the user-supplied image.
    """
    from flyteplugins.airflow.task import AirflowContainerTask  # applies patches
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    custom_image = Image.from_debian_base().with_pip_packages("apache-airflow<3.0.0")
    env = flyte.TaskEnvironment(name="test-dag-env", image=custom_image)

    with DAG(dag_id="test_image_preserved", flyte_env=env) as dag:
        BashOperator(task_id="say_hello", bash_command='echo hello')

    assert dag.flyte_task is not None
    parent_env = dag.flyte_task.parent_env()
    assert parent_env is not None, "flyte_task must be attached to an environment"
    assert parent_env.image is custom_image, (
        f"Expected the user-supplied image to be preserved, got: {parent_env.image}"
    )

    # Operator tasks must also inherit the env's image so they can be
    # serialized correctly when submitted as sub-tasks during remote execution.
    for task_name, op_task in parent_env.tasks.items():
        if task_name != f"{env.name}.dag_test_image_preserved":
            assert op_task.image is custom_image, (
                f"Operator task {task_name!r} did not inherit env image: {op_task.image}"
            )


def test_operator_tasks_registered_in_env(monkeypatch):
    """Operator tasks must appear in env.tasks so they are included in deployment."""
    from flyteplugins.airflow.task import AirflowContainerTask  # applies patches
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    env = flyte.TaskEnvironment(name="test-dag-tasks-env")

    with DAG(dag_id="test_tasks_registered", flyte_env=env) as dag:
        BashOperator(task_id="step1", bash_command='echo step1')
        BashOperator(task_id="step2", bash_command='echo step2')

    parent_env = dag.flyte_task.parent_env()
    # Both operator tasks and the orchestrator task must be in env.tasks.
    env_task_names = list(parent_env.tasks.keys())
    assert any("step1" in name for name in env_task_names), (
        f"step1 not found in env.tasks: {env_task_names}"
    )
    assert any("step2" in name for name in env_task_names), (
        f"step2 not found in env.tasks: {env_task_names}"
    )


def test_default_env_created_when_flyte_env_omitted():
    """When flyte_env is not supplied, a default TaskEnvironment is created using
    the dag_id as the name and a Debian-base image with airflow packages."""
    from flyteplugins.airflow.task import AirflowContainerTask  # applies patches
    from airflow import DAG
    from airflow.operators.bash import BashOperator

    with DAG(dag_id="test_default_env") as dag:
        BashOperator(task_id="greet", bash_command='echo hi')

    assert dag.flyte_task is not None
    parent_env = dag.flyte_task.parent_env()
    assert parent_env is not None
    assert parent_env.name == "test_default_env"
    # Default env should have a real image (not None or "auto")
    assert isinstance(parent_env.image, Image)

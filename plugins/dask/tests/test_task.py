import pathlib

import flyte
import pytest
from flyte import Resources
from flyte.models import SerializationContext

from flyteplugins.dask import Dask, Scheduler, WorkerGroup


@pytest.fixture
def serialization_context() -> SerializationContext:
    return SerializationContext(
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


def test_dask_task_with_default_config(serialization_context: SerializationContext):
    dask_config = Dask()
    dask_env = flyte.TaskEnvironment(
        name="dask_env",
        plugin_config=dask_config,
    )

    @dask_env.task()
    def dask_task():
        pass

    assert dask_task.plugin_config == dask_config
    assert dask_task.task_type == "dask"

    expected_dict = {
        "scheduler": {},
        "workers": {
            "numberOfWorkers": 1,
        },
    }

    assert dask_task.custom_config(serialization_context) == expected_dict


def test_dask_task_get_custom(serialization_context: SerializationContext):
    dask_config = Dask(
        scheduler=Scheduler(
            image="scheduler:latest",
            resources=Resources(cpu=("1", "1"), memory=("1Gi", "1Gi")),
        ),
        workers=WorkerGroup(
            number_of_workers=123,
            image="dask_cluster:latest",
            resources=Resources(cpu="1", memory="1Gi"),
        ),
    )
    dask_env = flyte.TaskEnvironment(
        name="dask_env",
        plugin_config=dask_config,
    )

    @dask_env.task()
    def dask_task():
        pass

    expected_custom_dict = {
        "scheduler": {
            "image": "scheduler:latest",
            "resources": {
                "requests": [{"name": "CPU", "value": "1"}, {"name": "MEMORY", "value": "1Gi"}],
                "limits": [{"name": "CPU", "value": "1"}, {"name": "MEMORY", "value": "1Gi"}],
            },
        },
        "workers": {
            "numberOfWorkers": 123,
            "image": "dask_cluster:latest",
            "resources": {
                "requests": [{"name": "CPU", "value": "1"}, {"name": "MEMORY", "value": "1Gi"}],
            },
        },
    }
    custom_dict = dask_task.custom_config(serialization_context)
    assert custom_dict == expected_custom_dict

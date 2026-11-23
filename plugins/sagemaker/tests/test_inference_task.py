import pytest

from flyteplugins.sagemaker import (
    SageMakerDeleteEndpointConfigTask,
    SageMakerDeleteEndpointTask,
    SageMakerDeleteModelTask,
    SageMakerEndpointConfigTask,
    SageMakerEndpointTask,
    SageMakerInvokeEndpointTask,
    SageMakerModelTask,
)


@pytest.mark.parametrize(
    ("task_cls", "method", "service"),
    [
        (SageMakerModelTask, "create_model", "sagemaker"),
        (SageMakerEndpointConfigTask, "create_endpoint_config", "sagemaker"),
        (SageMakerDeleteEndpointTask, "delete_endpoint", "sagemaker"),
        (SageMakerDeleteEndpointConfigTask, "delete_endpoint_config", "sagemaker"),
        (SageMakerDeleteModelTask, "delete_model", "sagemaker"),
        (SageMakerInvokeEndpointTask, "invoke_endpoint_async", "sagemaker-runtime"),
    ],
)
def test_sagemaker_boto_task(task_cls, method, service):
    task = task_cls(name=method, config={"Name": "{inputs.name}"}, region="us-west-2", inputs={"name": str})
    custom = task.custom_config(None)
    assert custom["method"] == method
    assert custom["service"] == service
    assert set(task.interface.inputs) == {"name"}


def test_endpoint_task():
    task = SageMakerEndpointTask(
        name="endpoint",
        config={"EndpointName": "demo"},
        region="us-west-2",
    )
    assert task.task_type == "sagemaker-endpoint"
    assert task.custom_config(None)["config"]["EndpointName"] == "demo"


def test_model_task_supports_images():
    task = SageMakerModelTask(
        name="model",
        config={"PrimaryContainer": {"Image": "{images.primary}"}},
        region="us-west-2",
        images={"primary": "repo/image"},
    )
    assert task.custom_config(None)["images"] == {"primary": "repo/image"}

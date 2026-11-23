import pytest

from flyteplugins.sagemaker import BotoConfig, BotoTask


def test_boto_task_and_config():
    task = BotoTask(
        name="model",
        task_config=BotoConfig(
            service="sagemaker",
            method="create_model",
            config={"ModelName": "model-{idempotence_token}"},
            region="us-west-2",
        ),
        inputs={"name": str},
    )
    assert task.task_type == "boto"
    assert set(task.interface.outputs) == {"result", "idempotence_token"}
    assert task.custom_config(None)["method"] == "create_model"


@pytest.mark.parametrize(
    ("service", "method", "message"),
    [
        ("", "create_model", "service is required"),
        ("sagemaker", "", "public boto3"),
        ("sagemaker", "_private", "public boto3"),
    ],
)
def test_boto_config_validation(service, method, message):
    with pytest.raises(ValueError, match=message):
        BotoConfig(service=service, method=method, config={})


def test_boto_config_preserves_images():
    config = BotoConfig(
        service="sagemaker",
        method="create_model",
        config={},
        images={"primary": "repo/image:tag"},
    )
    assert config.to_custom_config()["images"] == {"primary": "repo/image:tag"}

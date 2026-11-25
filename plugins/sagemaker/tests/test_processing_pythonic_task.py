import flyte
import pytest

from flyteplugins.sagemaker import SageMakerProcessing, SageMakerProcessingTask


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution_role_arn": "", "region": "us-west-2"}, "execution_role_arn"),
        ({"execution_role_arn": "role", "region": ""}, "region"),
        ({"execution_role_arn": "role", "region": "us-west-2", "volume_size_in_gb": 0}, "volume_size"),
        ({"execution_role_arn": "role", "region": "us-west-2", "max_runtime_in_seconds": 0}, "max_runtime"),
    ],
)
def test_dataclass_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SageMakerProcessing(**kwargs)


def test_to_from_dict_round_trip():
    config = SageMakerProcessing(
        execution_role_arn="role",
        region="us-west-2",
        network_config={"VpcConfig": {"Subnets": ["subnet-1"]}},
    )
    assert SageMakerProcessing.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("image", ["repo/image:tag", "repo/image:v2"])
def test_string_image_preserved(image):
    config = SageMakerProcessing(execution_role_arn="role", region="us-west-2")
    env = flyte.TaskEnvironment(name=f"processing-{image.rsplit(':', 1)[1]}", plugin_config=config, image=image)

    @env.task
    async def process() -> int:
        return 1

    assert isinstance(process, SageMakerProcessingTask)
    assert process.image.uri == image
    assert process.task_type == "sagemaker-processing-task"


def test_get_custom_returns_config_dict():
    config = SageMakerProcessing(execution_role_arn="role", region="us-west-2", tags={"team": "ml"})
    env = flyte.TaskEnvironment(name="processing-custom", plugin_config=config, image="repo/image")

    @env.task
    async def process() -> int:
        return 1

    assert process.custom_config(None)["tags"] == {"team": "ml"}

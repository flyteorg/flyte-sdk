import flyte
import pytest

from flyteplugins.sagemaker import SageMakerTraining, SageMakerTrainingTask


def test_dataclass_validation():
    with pytest.raises(ValueError, match="execution_role_arn"):
        SageMakerTraining(execution_role_arn="", region="us-west-2")
    with pytest.raises(ValueError, match="instance_count=1"):
        SageMakerTraining(execution_role_arn="role", region="us-west-2", instance_count=2)


def test_to_from_dict_round_trip_with_output_path():
    config = SageMakerTraining(
        execution_role_arn="role",
        region="us-west-2",
        output_s3_path="s3://bucket/models",
        vpc_config={"Subnets": ["subnet-1"], "SecurityGroupIds": ["sg-1"]},
    )
    assert SageMakerTraining.from_dict(config.to_dict()) == config


def test_output_s3_path_optional_and_dropped_when_none():
    config = SageMakerTraining(execution_role_arn="role", region="us-west-2")
    assert "output_s3_path" not in config.to_dict()


@pytest.mark.parametrize("image", ["repo/image:tag", "repo/other:tag"])
def test_explicit_container_image_is_applied(image):
    config = SageMakerTraining(execution_role_arn="role", region="us-west-2", container_image=image)
    env = flyte.TaskEnvironment(name=f"training-{image.split('/')[1].split(':')[0]}", plugin_config=config)

    @env.task
    async def train() -> int:
        return 1

    assert isinstance(train, SageMakerTrainingTask)
    assert train.task_type == "sagemaker-training-task"
    assert train.image.uri == image


def test_get_custom_returns_config_dict():
    config = SageMakerTraining(execution_role_arn="role", region="us-west-2")
    env = flyte.TaskEnvironment(name="training-custom", plugin_config=config, image="repo/image:tag")

    @env.task
    async def train() -> int:
        return 1

    assert train.custom_config(None)["execution_role_arn"] == "role"
    assert "container_image" not in train.custom_config(None)

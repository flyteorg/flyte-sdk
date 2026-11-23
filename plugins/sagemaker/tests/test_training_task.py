import pytest

from flyteplugins.sagemaker import (
    SageMakerDescribeTrainingJobTask,
    SageMakerStopTrainingJobTask,
    SageMakerTrainingJob,
    SageMakerTrainingJobTask,
)


def test_training_job_task_interface_and_custom():
    task = SageMakerTrainingJobTask(
        name="train",
        config={"TrainingJobName": "job", "ResourceConfig": {"InstanceCount": 1}},
        region="us-west-2",
        inputs={"dataset": str},
    )
    assert task.task_type == "sagemaker-training-job"
    assert task.custom_config(None)["config"]["ResourceConfig"]["InstanceCount"] == 1
    assert set(task.interface.inputs) == {"dataset"}


@pytest.mark.parametrize(
    ("task_cls", "method"),
    [
        (SageMakerStopTrainingJobTask, "stop_training_job"),
        (SageMakerDescribeTrainingJobTask, "describe_training_job"),
    ],
)
def test_helper_boto_tasks_use_correct_method(task_cls, method):
    task = task_cls(name=method, config={"TrainingJobName": "job"}, region="us-west-2")
    assert task.custom_config(None)["method"] == method


def test_decorator_config_defaults_job_name():
    config = SageMakerTrainingJob(request={"RoleArn": "role"}, region="us-west-2")
    assert config.to_custom_config("train")["request"]["TrainingJobName"] == "train"


def test_decorator_config_requires_request():
    with pytest.raises(ValueError, match="request is required"):
        SageMakerTrainingJob(request={})

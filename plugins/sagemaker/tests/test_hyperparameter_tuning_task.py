import pytest

from flyteplugins.sagemaker import (
    SageMakerDescribeHyperParameterTuningJobTask,
    SageMakerHyperParameterTuningJob,
    SageMakerHyperParameterTuningJobTask,
    SageMakerStopHyperParameterTuningJobTask,
)


def test_hyperparameter_tuning_job_task_interface_and_custom():
    task = SageMakerHyperParameterTuningJobTask(
        name="tune",
        config={"HyperParameterTuningJobName": "job"},
        region="us-west-2",
        images={"training": "repo/train"},
    )
    assert task.task_type == "sagemaker-hyperparameter-tuning-job"
    assert task.custom_config(None)["images"]["training"] == "repo/train"


@pytest.mark.parametrize(
    ("task_cls", "method"),
    [
        (SageMakerStopHyperParameterTuningJobTask, "stop_hyper_parameter_tuning_job"),
        (SageMakerDescribeHyperParameterTuningJobTask, "describe_hyper_parameter_tuning_job"),
    ],
)
def test_helper_boto_tasks_use_correct_method(task_cls, method):
    assert task_cls(name="x", config={}, region="us-west-2").custom_config(None)["method"] == method


@pytest.mark.parametrize("region", [None, "us-east-1"])
def test_decorator_config(region):
    config = SageMakerHyperParameterTuningJob(request={"RoleArn": "role"}, region=region)
    assert config.to_custom_config("tune")["region"] == region

import pytest

from flyteplugins.sagemaker import (
    SageMakerDescribeProcessingJobTask,
    SageMakerProcessingJob,
    SageMakerProcessingJobTask,
    SageMakerStopProcessingJobTask,
)


def test_processing_job_task_interface_and_custom():
    task = SageMakerProcessingJobTask(
        name="process",
        config={"ProcessingJobName": "job"},
        region="us-west-2",
        images={"main": "repo/image"},
    )
    assert task.task_type == "sagemaker-processing-job"
    assert task.custom_config(None)["images"]["main"] == "repo/image"


@pytest.mark.parametrize(
    ("task_cls", "method"),
    [
        (SageMakerStopProcessingJobTask, "stop_processing_job"),
        (SageMakerDescribeProcessingJobTask, "describe_processing_job"),
    ],
)
def test_helper_boto_tasks_use_correct_method(task_cls, method):
    assert task_cls(name="x", config={}, region="us-west-2").custom_config(None)["method"] == method


def test_processing_config_job_name():
    custom = SageMakerProcessingJob(request={"RoleArn": "role"}).to_custom_config("process")
    assert custom["request"]["ProcessingJobName"] == "process"


def test_processing_config_preserves_explicit_name():
    custom = SageMakerProcessingJob(request={"ProcessingJobName": "fixed"}).to_custom_config("other")
    assert custom["request"]["ProcessingJobName"] == "fixed"

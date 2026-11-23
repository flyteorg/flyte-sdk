import pytest

from flyteplugins.sagemaker import (
    SageMakerDescribeTransformJobTask,
    SageMakerStopTransformJobTask,
    SageMakerTransformJob,
    SageMakerTransformJobTask,
)


def test_transform_job_task_interface_and_custom():
    task = SageMakerTransformJobTask(
        name="transform",
        config={"TransformJobName": "job"},
        region="us-west-2",
    )
    assert task.task_type == "sagemaker-transform-job"
    assert task.custom_config(None)["config"]["TransformJobName"] == "job"


@pytest.mark.parametrize(
    ("task_cls", "method"),
    [
        (SageMakerStopTransformJobTask, "stop_transform_job"),
        (SageMakerDescribeTransformJobTask, "describe_transform_job"),
    ],
)
def test_helper_boto_tasks_use_correct_method(task_cls, method):
    assert task_cls(name="x", config={}, region="us-west-2").custom_config(None)["method"] == method


@pytest.mark.parametrize("explicit", [True, False])
def test_transform_config_name(explicit):
    request = {"TransformJobName": "fixed"} if explicit else {"ModelName": "model"}
    name = SageMakerTransformJob(request=request).to_custom_config("generated")["request"]["TransformJobName"]
    assert name == ("fixed" if explicit else "generated")

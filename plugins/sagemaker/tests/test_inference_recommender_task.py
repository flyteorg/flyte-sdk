import pytest

from flyteplugins.sagemaker import (
    SageMakerDescribeInferenceRecommenderJobTask,
    SageMakerInferenceRecommenderJob,
    SageMakerInferenceRecommenderJobTask,
    SageMakerStopInferenceRecommenderJobTask,
)


def test_inference_recommender_job_task_interface_and_custom():
    task = SageMakerInferenceRecommenderJobTask(
        name="recommend",
        config={"JobName": "job"},
        region="us-west-2",
    )
    assert task.task_type == "sagemaker-inference-recommender-job"
    assert task.custom_config(None)["config"]["JobName"] == "job"


@pytest.mark.parametrize(
    ("task_cls", "method"),
    [
        (SageMakerStopInferenceRecommenderJobTask, "stop_inference_recommendations_job"),
        (SageMakerDescribeInferenceRecommenderJobTask, "describe_inference_recommendations_job"),
    ],
)
def test_helper_boto_tasks_use_correct_method(task_cls, method):
    assert task_cls(name="x", config={}, region="us-west-2").custom_config(None)["method"] == method


@pytest.mark.parametrize("job_request", [{"JobType": "Default"}, {"JobName": "fixed"}])
def test_decorator_config(job_request):
    custom = SageMakerInferenceRecommenderJob(request=job_request).to_custom_config("recommend")
    assert custom["request"]["JobName"] == job_request.get("JobName", "recommend")

import pytest

from flyteplugins.sagemaker.task import (
    SageMakerProcessingJob,
    SageMakerTrainingJob,
    SageMakerTransformJob,
)


@pytest.mark.parametrize(
    ("config_class", "name_key"),
    [
        (SageMakerTrainingJob, "TrainingJobName"),
        (SageMakerProcessingJob, "ProcessingJobName"),
        (SageMakerTransformJob, "TransformJobName"),
    ],
)
def test_managed_job_serializes_exact_boto_request(
    config_class: type,
    name_key: str,
) -> None:
    config = config_class(
        region="us-west-2",
        request={
            "RoleArn": "arn:aws:iam::123456789012:role/sagemaker-job",
            "ResourceConfig": {
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
        },
    )

    assert config.to_custom_config("example-job") == {
        "region": "us-west-2",
        "request": {
            name_key: "example-job",
            "RoleArn": "arn:aws:iam::123456789012:role/sagemaker-job",
            "ResourceConfig": {
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
        },
    }


@pytest.mark.parametrize(
    "config_class",
    [
        SageMakerTrainingJob,
        SageMakerProcessingJob,
        SageMakerTransformJob,
    ],
)
def test_managed_job_requires_request(config_class: type) -> None:
    with pytest.raises(ValueError, match="request is required"):
        config_class(request={})

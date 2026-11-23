import pytest

from flyteplugins.emr_serverless.task import EMRServerless


def test_emr_serverless_serializes_start_job_run_request() -> None:
    config = EMRServerless(
        application_id="app-123",
        execution_role_arn="arn:aws:iam::123456789012:role/emr-job",
        job_driver={
            "sparkSubmit": {
                "entryPoint": "s3://example-bucket/jobs/main.py",
            }
        },
        region="us-west-2",
        tags={"purpose": "integration-test"},
        execution_timeout_minutes=15,
    )

    assert config.to_custom_config("example-task") == {
        "region": "us-west-2",
        "request": {
            "applicationId": "app-123",
            "executionRoleArn": "arn:aws:iam::123456789012:role/emr-job",
            "jobDriver": {
                "sparkSubmit": {
                    "entryPoint": "s3://example-bucket/jobs/main.py",
                }
            },
            "executionTimeoutMinutes": 15,
            "name": "example-task",
            "tags": {"purpose": "integration-test"},
        },
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"application_id": ""},
        {"execution_role_arn": ""},
        {"job_driver": {}},
        {
            "job_driver": {
                "sparkSubmit": {},
                "hive": {},
            }
        },
        {"execution_timeout_minutes": 0},
    ],
)
def test_emr_serverless_rejects_invalid_configuration(overrides: dict) -> None:
    values = {
        "application_id": "app-123",
        "execution_role_arn": "arn:aws:iam::123456789012:role/emr-job",
        "job_driver": {
            "sparkSubmit": {
                "entryPoint": "s3://example-bucket/jobs/main.py",
            }
        },
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        EMRServerless(**values)

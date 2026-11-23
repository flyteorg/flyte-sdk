# AWS SageMaker Managed Jobs Plugin for Flyte

Run SageMaker training, processing, and batch-transform jobs through Flyte's
async connector service. The plugin preserves the boto3 request shape, reports
provider state and failure reasons, exposes AWS Console links, returns stable
result dictionaries, and stops active jobs when a Flyte action is aborted.

## Installation

```bash
pip install flyteplugins-sagemaker
```

Install the package in the `flyteconnector` image and the task-authoring
environment. Route these task types to the connector service:

```text
sagemaker-training-job
sagemaker-processing-job
sagemaker-transform-job
```

The connector uses boto3's standard credential chain. The connector identity
needs permission to create, describe, and stop only the SageMaker job types
that the platform enables, plus `iam:PassRole` for approved execution roles.

## Training

```python
import flyte
from flyteplugins.sagemaker import SageMakerTrainingJob

env = flyte.TaskEnvironment(
    name="training",
    plugin_config=SageMakerTrainingJob(
        region="us-west-2",
        request={
            "RoleArn": "arn:aws:iam::123456789012:role/sagemaker-job",
            "AlgorithmSpecification": {
                "TrainingImage": "123456789012.dkr.ecr.us-west-2.amazonaws.com/trainer:latest",
                "TrainingInputMode": "File",
            },
            "ResourceConfig": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            "InputDataConfig": [],
            "OutputDataConfig": {
                "S3OutputPath": "s3://example-bucket/training-output",
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 600,
            },
        },
    ),
)


@env.task
async def train() -> dict:
    """Return stable training-job result metadata."""
```

The decorated function declares the Flyte task interface; managed mode submits
the supplied boto3 request and does not execute the function body. Pythonic
function execution inside the SageMaker container is a separate mode and is
intentionally outside this first contribution.

When the boto request omits `TrainingJobName`, the serialized Flyte task name
is used. Explicit names are preserved, which allows workflows to implement
their own naming and idempotency policy.

Processing and transform tasks use the same pattern with
`SageMakerProcessingJob` and `SageMakerTransformJob`. Their request dictionaries
are passed to `CreateProcessingJob` and `CreateTransformJob` respectively.

## Outputs

A managed job may declare no Flyte output or one dictionary output. Successful
jobs return a stable provider-neutral envelope:

```python
{
    "job_name": "example-job",
    "status": "Completed",
    # Job-specific result fields when present:
    "model_artifacts": {...},
    "processing_output_config": {...},
    "transform_output": {...},
}
```

Timestamps, decimal values, byte strings, and nested provider structures are
converted to JSON-safe values before being returned.

## Retry and cancellation behavior

- Provider throttling and transient service failures return
  `RETRYABLE_FAILED`.
- Duplicate create responses are accepted only after the corresponding
  describe call confirms that the named job exists.
- Completed, failed, and stopped jobs are not stopped again.
- Missing jobs are treated as already deleted.

## Current scope

This plugin submits managed boto3 job requests. Python function execution
inside SageMaker containers, HyperPod configuration, model/endpoint operations,
and a generic boto connector are separate follow-ups so each API and security
boundary can be reviewed independently.

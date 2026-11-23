# AWS SageMaker Managed Jobs Plugin for Flyte

Run the public Flytekit SageMaker feature set through Flyte 2's connector
service. The plugin includes managed jobs, Pythonic training and processing,
generic boto calls, endpoint deployment helpers, stable result projections,
typed task returns, and cancellation.

## Installation

```bash
pip install flyteplugins-sagemaker
```

Install the package in the `flyteconnector` image and the task-authoring
environment. Route these task types to the connector service:

```text
sagemaker-training-job
sagemaker-training-task
sagemaker-processing-job
sagemaker-processing-task
sagemaker-transform-job
sagemaker-hyperparameter-tuning-job
sagemaker-inference-recommender-job
sagemaker-endpoint
boto
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
the supplied boto3 request and does not execute the function body. Use
`SageMakerTraining` or `SageMakerProcessing` as the plugin config to run the
function itself in SageMaker. Pythonic mode uses the rendered Flyte 2 `a0`
entrypoint and reads `outputs.pb` / `error.pb` from the action output prefix.

When the boto request omits `TrainingJobName`, the serialized Flyte task name
is used. Explicit names are preserved, which allows workflows to implement
their own naming and idempotency policy.

Processing and transform tasks use the same pattern with
`SageMakerProcessingJob` and `SageMakerTransformJob`. Their request dictionaries
are passed to `CreateProcessingJob` and `CreateTransformJob` respectively.

## Public parity matrix

| Public Flytekit surface | Flyte 2 export | Task type | Test family |
| --- | --- | --- | --- |
| Generic boto config/task/connector | `BotoConfig`, `BotoTask`, `BotoConnector` | `boto` | `test_boto3_task`, `test_boto3_mixin`, `test_boto3_connector` |
| Training managed job | `SageMakerTrainingJobTask`, `SageMakerTrainingJobConnector` | `sagemaker-training-job` | `test_training_task`, `test_training_connector` |
| Training Pythonic mode | `SageMakerTraining`, `SageMakerTrainingTask`, `SageMakerTrainingTaskConnector` | `sagemaker-training-task` | `test_training_pythonic_task`, `test_pythonic_connector` |
| Processing managed job | `SageMakerProcessingJobTask`, `SageMakerProcessingJobConnector` | `sagemaker-processing-job` | `test_processing_task`, `test_processing_connector` |
| Processing Pythonic mode | `SageMakerProcessing`, `SageMakerProcessingTask`, `SageMakerProcessingTaskConnector` | `sagemaker-processing-task` | `test_processing_pythonic_task`, `test_pythonic_connector` |
| Batch transform | `SageMakerTransformJobTask`, `SageMakerTransformJobConnector` | `sagemaker-transform-job` | `test_batch_transform_task`, `test_batch_transform_connector` |
| Hyperparameter tuning | `SageMakerHyperParameterTuningJobTask`, `SageMakerHyperParameterTuningJobConnector` | `sagemaker-hyperparameter-tuning-job` | `test_hyperparameter_tuning_task`, `test_hyperparameter_tuning_connector` |
| Inference recommender | `SageMakerInferenceRecommenderJobTask`, `SageMakerInferenceRecommenderJobConnector` | `sagemaker-inference-recommender-job` | `test_inference_recommender_task`, `test_inference_recommender_connector` |
| Endpoint lifecycle | model/config/endpoint/invoke/delete task exports, `SageMakerEndpointConnector` | `boto`, `sagemaker-endpoint` | `test_inference_task`, `test_inference_connector` |
| Deployment composition | `create_sagemaker_deployment`, `delete_sagemaker_deployment` | composed task calls | `test_inference_workflow` |
| Triton URI template | `triton_image_uri` | n/a | `test_boto3_mixin` |

Every managed job also exports matching stop and describe helpers. Result
dictionaries retain the curated public Flytekit PascalCase keys, such as
`ModelArtifacts.S3ModelArtifacts` and `TransformOutput.S3OutputPath`.

## Retry and cancellation behavior

- Provider throttling and transient service failures return
  `RETRYABLE_FAILED`.
- Duplicate create responses are accepted only after the corresponding
  describe call confirms that the named job exists.
- Completed, failed, and stopped jobs are not stopped again.
- Missing jobs are treated as already deleted.

## Generic request rendering

`BotoConfig.config` and managed task configs support `{inputs.name}`,
`{images.name}`, and `{idempotence_token}` placeholders. Whole-number floats
are recursively normalized before stable hashing and boto invocation. List
order remains significant. Triton URI templates are resolved for the runtime
region, including AWS China partitions.

HyperPod is intentionally excluded because it is not part of the public
Flytekit SageMaker plugin baseline.

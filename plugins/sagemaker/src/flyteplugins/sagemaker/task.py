from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate
from flyte.models import SerializationContext

TRAINING_TASK_TYPE = "sagemaker-training-job"
PROCESSING_TASK_TYPE = "sagemaker-processing-job"
TRANSFORM_TASK_TYPE = "sagemaker-transform-job"


@dataclass
class _ManagedJob:
    request: Dict[str, Any]
    region: Optional[str] = None

    job_name_key: ClassVar[str] = ""

    def __post_init__(self) -> None:
        if not self.request:
            raise ValueError("request is required")

    def to_custom_config(self, task_name: str) -> Dict[str, Any]:
        request = dict(self.request)
        request.setdefault(self.job_name_key, task_name)
        return {
            "region": self.region,
            "request": request,
        }


@dataclass
class SageMakerTrainingJob(_ManagedJob):
    """Configuration for a SageMaker `CreateTrainingJob` request.

    Args:
        request: Exact boto3 `CreateTrainingJob` keyword arguments.
        region: AWS region. When omitted, boto3's default resolution applies.
    """

    job_name_key = "TrainingJobName"


@dataclass
class SageMakerProcessingJob(_ManagedJob):
    """Configuration for a SageMaker `CreateProcessingJob` request.

    Args:
        request: Exact boto3 `CreateProcessingJob` keyword arguments.
        region: AWS region. When omitted, boto3's default resolution applies.
    """

    job_name_key = "ProcessingJobName"


@dataclass
class SageMakerTransformJob(_ManagedJob):
    """Configuration for a SageMaker `CreateTransformJob` request.

    Args:
        request: Exact boto3 `CreateTransformJob` keyword arguments.
        region: AWS region. When omitted, boto3's default resolution applies.
    """

    job_name_key = "TransformJobName"


class _ManagedJobFunctionTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    plugin_config: _ManagedJob
    task_type_name: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_type = self.task_type_name

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        """Return the serialized SageMaker API request."""
        if len(self.interface.outputs) > 1:
            raise ValueError("SageMaker managed jobs support at most one Flyte output")
        return self.plugin_config.to_custom_config(self.name)


class SageMakerTrainingFunctionTask(_ManagedJobFunctionTask):
    task_type_name = TRAINING_TASK_TYPE


class SageMakerProcessingFunctionTask(_ManagedJobFunctionTask):
    task_type_name = PROCESSING_TASK_TYPE


class SageMakerTransformFunctionTask(_ManagedJobFunctionTask):
    task_type_name = TRANSFORM_TASK_TYPE


TaskPluginRegistry.register(SageMakerTrainingJob, SageMakerTrainingFunctionTask)
TaskPluginRegistry.register(SageMakerProcessingJob, SageMakerProcessingFunctionTask)
TaskPluginRegistry.register(SageMakerTransformJob, SageMakerTransformFunctionTask)

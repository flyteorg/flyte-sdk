from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Type

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import NativeInterface, SerializationContext

TRAINING_TASK_TYPE = "sagemaker-training-job"
TRAINING_PYTHON_TASK_TYPE = "sagemaker-training-task"
PROCESSING_TASK_TYPE = "sagemaker-processing-job"
PROCESSING_PYTHON_TASK_TYPE = "sagemaker-processing-task"
TRANSFORM_TASK_TYPE = "sagemaker-transform-job"
HYPERPARAMETER_TUNING_TASK_TYPE = "sagemaker-hyperparameter-tuning-job"
INFERENCE_RECOMMENDER_TASK_TYPE = "sagemaker-inference-recommender-job"
ENDPOINT_TASK_TYPE = "sagemaker-endpoint"
BOTO_TASK_TYPE = "boto"


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


@dataclass
class SageMakerHyperParameterTuningJob(_ManagedJob):
    """Configuration for a SageMaker `CreateHyperParameterTuningJob` request."""

    job_name_key = "HyperParameterTuningJobName"


@dataclass
class SageMakerInferenceRecommenderJob(_ManagedJob):
    """Configuration for a SageMaker `CreateInferenceRecommendationsJob` request."""

    job_name_key = "JobName"


@dataclass
class SageMakerEndpoint(_ManagedJob):
    """Configuration for a SageMaker `CreateEndpoint` request."""

    job_name_key = "EndpointName"


@dataclass
class BotoConfig:
    """Configuration for one generic boto3 API call."""

    service: str
    method: str
    config: Dict[str, Any]
    region: Optional[str] = None
    images: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.service:
            raise ValueError("service is required")
        if not self.method or self.method.startswith("_"):
            raise ValueError("method must name a public boto3 client operation")

    def to_custom_config(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "method": self.method,
            "config": dict(self.config),
            "region": self.region,
            "images": _serialized_images(self.images),
        }


# Flyte 2's plugin-config spelling. BotoConfig remains exported for public
# Flytekit source compatibility.
Boto = BotoConfig


def _serialized_images(images: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not images:
        return None
    serialized = {}
    for name, image in images.items():
        value = getattr(image, "uri", image)
        if not isinstance(value, str):
            raise TypeError(f"Image {name!r} must be a URI string or flyte.Image")
        serialized[name] = value
    return serialized


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


class SageMakerHyperParameterTuningFunctionTask(_ManagedJobFunctionTask):
    task_type_name = HYPERPARAMETER_TUNING_TASK_TYPE


class SageMakerInferenceRecommenderFunctionTask(_ManagedJobFunctionTask):
    task_type_name = INFERENCE_RECOMMENDER_TASK_TYPE


class SageMakerEndpointFunctionTask(_ManagedJobFunctionTask):
    task_type_name = ENDPOINT_TASK_TYPE


class BotoFunctionTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    """Flyte 2 function-task adapter for `BotoConfig`."""

    plugin_config: BotoConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_type = BOTO_TASK_TYPE

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        return self.plugin_config.to_custom_config()


def _interface(inputs: Optional[Dict[str, Type]], outputs: Optional[Dict[str, Type]] = None) -> NativeInterface:
    return NativeInterface.from_types(
        inputs={name: (value, inspect.Parameter.empty) for name, value in (inputs or {}).items()},
        outputs=outputs or {"result": dict},
    )


class BotoTask(AsyncConnectorExecutorMixin, TaskTemplate):
    """Standalone generic boto task compatible with the public Flytekit API."""

    def __init__(
        self,
        name: str,
        task_config: BotoConfig,
        inputs: Optional[Dict[str, Type]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            interface=_interface(inputs, {"result": dict, "idempotence_token": str}),
            task_type=BOTO_TASK_TYPE,
            **kwargs,
        )
        self.task_config = task_config
        self.runs_in_container = False

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        custom = self.task_config.to_custom_config()
        custom["images"] = _serialized_images(self.task_config.images)
        return custom


class _ManagedStandaloneTask(AsyncConnectorExecutorMixin, TaskTemplate):
    task_type_name: ClassVar[str]

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        region: Optional[str] = None,
        images: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Type]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            interface=_interface(inputs),
            task_type=self.task_type_name,
            **kwargs,
        )
        self._config = config
        self._region = region
        self._images = images
        self.runs_in_container = False

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        return {"config": self._config, "region": self._region, "images": _serialized_images(self._images)}


class SageMakerTrainingJobTask(_ManagedStandaloneTask):
    task_type_name = TRAINING_TASK_TYPE


class SageMakerProcessingJobTask(_ManagedStandaloneTask):
    task_type_name = PROCESSING_TASK_TYPE


class SageMakerTransformJobTask(_ManagedStandaloneTask):
    task_type_name = TRANSFORM_TASK_TYPE


class SageMakerHyperParameterTuningJobTask(_ManagedStandaloneTask):
    task_type_name = HYPERPARAMETER_TUNING_TASK_TYPE


class SageMakerInferenceRecommenderJobTask(_ManagedStandaloneTask):
    task_type_name = INFERENCE_RECOMMENDER_TASK_TYPE


class SageMakerEndpointTask(_ManagedStandaloneTask):
    task_type_name = ENDPOINT_TASK_TYPE


def _boto_helper(class_name: str, method: str, service: str = "sagemaker") -> type[BotoTask]:
    class Helper(BotoTask):
        def __init__(
            self,
            name: str,
            config: Dict[str, Any],
            region: Optional[str] = None,
            inputs: Optional[Dict[str, Type]] = None,
            images: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                name=name,
                task_config=BotoConfig(
                    service=service,
                    method=method,
                    config=config,
                    region=region,
                    images=images,
                ),
                inputs=inputs,
                **kwargs,
            )

    Helper.__name__ = class_name
    Helper.__qualname__ = class_name
    return Helper


SageMakerStopTrainingJobTask = _boto_helper("SageMakerStopTrainingJobTask", "stop_training_job")
SageMakerDescribeTrainingJobTask = _boto_helper("SageMakerDescribeTrainingJobTask", "describe_training_job")
SageMakerStopProcessingJobTask = _boto_helper("SageMakerStopProcessingJobTask", "stop_processing_job")
SageMakerDescribeProcessingJobTask = _boto_helper("SageMakerDescribeProcessingJobTask", "describe_processing_job")
SageMakerStopTransformJobTask = _boto_helper("SageMakerStopTransformJobTask", "stop_transform_job")
SageMakerDescribeTransformJobTask = _boto_helper("SageMakerDescribeTransformJobTask", "describe_transform_job")
SageMakerStopHyperParameterTuningJobTask = _boto_helper(
    "SageMakerStopHyperParameterTuningJobTask", "stop_hyper_parameter_tuning_job"
)
SageMakerDescribeHyperParameterTuningJobTask = _boto_helper(
    "SageMakerDescribeHyperParameterTuningJobTask", "describe_hyper_parameter_tuning_job"
)
SageMakerStopInferenceRecommenderJobTask = _boto_helper(
    "SageMakerStopInferenceRecommenderJobTask", "stop_inference_recommendations_job"
)
SageMakerDescribeInferenceRecommenderJobTask = _boto_helper(
    "SageMakerDescribeInferenceRecommenderJobTask", "describe_inference_recommendations_job"
)
SageMakerModelTask = _boto_helper("SageMakerModelTask", "create_model")
SageMakerEndpointConfigTask = _boto_helper("SageMakerEndpointConfigTask", "create_endpoint_config")
SageMakerDeleteEndpointTask = _boto_helper("SageMakerDeleteEndpointTask", "delete_endpoint")
SageMakerDeleteEndpointConfigTask = _boto_helper("SageMakerDeleteEndpointConfigTask", "delete_endpoint_config")
SageMakerDeleteModelTask = _boto_helper("SageMakerDeleteModelTask", "delete_model")
SageMakerInvokeEndpointTask = _boto_helper(
    "SageMakerInvokeEndpointTask", "invoke_endpoint_async", service="sagemaker-runtime"
)


@dataclass
class PythonicJobConfig:
    """Shared configuration for running a Flyte task inside SageMaker."""

    execution_role_arn: str
    region: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    volume_size_in_gb: int = 30
    max_runtime_in_seconds: int = 3600
    environment: Optional[Dict[str, str]] = None
    kms_key_id: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    job_name_prefix: str = "flyte-"
    container_image: Optional[Any] = None

    def __post_init__(self) -> None:
        if not self.execution_role_arn:
            raise ValueError("execution_role_arn is required")
        if not self.region:
            raise ValueError("region is required")
        if self.instance_count != 1:
            raise ValueError("Pythonic SageMaker jobs currently require instance_count=1")
        if self.volume_size_in_gb < 1:
            raise ValueError("volume_size_in_gb must be at least 1")
        if self.max_runtime_in_seconds < 1:
            raise ValueError("max_runtime_in_seconds must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            item.name: value
            for item in dataclasses.fields(self)
            if item.name != "container_image" and (value := getattr(self, item.name)) is not None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PythonicJobConfig":
        names = {item.name for item in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


@dataclass
class SageMakerTraining(PythonicJobConfig):
    output_s3_path: Optional[str] = None
    vpc_config: Optional[Dict[str, Any]] = None


@dataclass
class SageMakerProcessing(PythonicJobConfig):
    network_config: Optional[Dict[str, Any]] = None


class _PythonicSageMakerTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    task_type_name: ClassVar[str]
    plugin_config: PythonicJobConfig

    def __post_init__(self) -> None:
        if self.plugin_config.container_image is not None:
            self.image = self.plugin_config.container_image
        if self.image is None:
            raise ValueError("Pythonic SageMaker jobs require a container image")
        super().__post_init__()
        self.task_type = self.task_type_name

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        custom = self.plugin_config.to_dict()
        custom.pop("container_image", None)
        return custom


class SageMakerTrainingTask(_PythonicSageMakerTask):
    task_type_name = TRAINING_PYTHON_TASK_TYPE


class SageMakerProcessingTask(_PythonicSageMakerTask):
    task_type_name = PROCESSING_PYTHON_TASK_TYPE


TaskPluginRegistry.register(SageMakerTrainingJob, SageMakerTrainingFunctionTask)
TaskPluginRegistry.register(SageMakerProcessingJob, SageMakerProcessingFunctionTask)
TaskPluginRegistry.register(SageMakerTransformJob, SageMakerTransformFunctionTask)
TaskPluginRegistry.register(SageMakerHyperParameterTuningJob, SageMakerHyperParameterTuningFunctionTask)
TaskPluginRegistry.register(SageMakerInferenceRecommenderJob, SageMakerInferenceRecommenderFunctionTask)
TaskPluginRegistry.register(SageMakerEndpoint, SageMakerEndpointFunctionTask)
TaskPluginRegistry.register(BotoConfig, BotoFunctionTask)
TaskPluginRegistry.register(SageMakerTraining, SageMakerTrainingTask)
TaskPluginRegistry.register(SageMakerProcessing, SageMakerProcessingTask)

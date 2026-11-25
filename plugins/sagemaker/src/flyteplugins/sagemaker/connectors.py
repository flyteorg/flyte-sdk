from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flyte import storage
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core import execution_pb2
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from flyteplugins.sagemaker._utils import (
    json_safe,
    normalize_integers,
    render_config,
    resolve_images,
    resolve_region,
)
from flyteplugins.sagemaker.task import (
    BOTO_TASK_TYPE,
    ENDPOINT_TASK_TYPE,
    HYPERPARAMETER_TUNING_TASK_TYPE,
    INFERENCE_RECOMMENDER_TASK_TYPE,
    PROCESSING_PYTHON_TASK_TYPE,
    PROCESSING_TASK_TYPE,
    TRAINING_PYTHON_TASK_TYPE,
    TRAINING_TASK_TYPE,
    TRANSFORM_TASK_TYPE,
)

_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "adaptive"})
_SERVICE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_METHOD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_DUPLICATE_CODES = {"ConflictException", "ResourceInUse"}
_MISSING_CODES = {"ResourceNotFound", "ResourceNotFoundException"}


class BotoCallError(RuntimeError):
    """Boto call failure carrying the rendered idempotence token."""

    def __init__(self, error: ClientError, idempotence_token: str):
        super().__init__(str(error))
        self.error = error
        self.idempotence_token = idempotence_token


class BotoCaller:
    service = "sagemaker"

    def _get_client(self, region: str) -> Any:
        """Compatibility hook for tests and subclasses that only use SageMaker."""
        return boto3.client(self.service, region_name=region, config=_BOTO_CONFIG)

    def _client(self, service: str, region: str) -> Any:
        if service == self.service:
            return self._get_client(region)
        return boto3.client(service, region_name=region, config=_BOTO_CONFIG)

    async def _call(
        self,
        method: str,
        config: Dict[str, Any],
        *,
        inputs: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, Any]] = None,
        region: Optional[str] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        service: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str]:
        final_region = resolve_region(region, inputs)
        final_images = resolve_images(images, final_region)
        request, token = render_config(
            config,
            inputs=inputs,
            images=final_images,
            region=final_region,
            metadata=task_execution_metadata,
        )
        client = self._client(service or self.service, final_region)
        operation = getattr(client, method, None)
        if operation is None or not callable(operation):
            raise ValueError(f"Boto3 client {service or self.service!r} has no public method {method!r}")
        try:
            response = await asyncio.to_thread(operation, **request)
        except ClientError as error:
            raise BotoCallError(error, token) from error
        response.pop("ResponseMetadata", None)
        return json_safe(response), token


def _error_parts(error: BotoCallError) -> tuple[str, str]:
    body = error.error.response.get("Error", {})
    return body.get("Code", ""), body.get("Message", "")


def _duplicate(error: BotoCallError) -> bool:
    code, message = _error_parts(error)
    return bool(error.idempotence_token) and (
        code in _DUPLICATE_CODES or (code == "ValidationException" and "Cannot create already existing" in message)
    )


def _terminal_stop_error(error: BotoCallError) -> bool:
    code, message = _error_parts(error)
    return code in _MISSING_CODES or (code == "ValidationException" and "non-running" in message)


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _training_outputs(response: Dict[str, Any]) -> Dict[str, Any]:
    metrics = [
        {
            "MetricName": item.get("MetricName"),
            "Value": item.get("Value"),
            "Timestamp": _iso(item.get("Timestamp")),
        }
        for item in response.get("FinalMetricDataList") or []
    ]
    return {
        "TrainingJobArn": response.get("TrainingJobArn"),
        "TrainingJobName": response.get("TrainingJobName"),
        "ModelArtifacts": {
            "S3ModelArtifacts": (response.get("ModelArtifacts") or {}).get("S3ModelArtifacts"),
        },
        "OutputDataConfig": {
            "S3OutputPath": (response.get("OutputDataConfig") or {}).get("S3OutputPath"),
        },
        "FinalMetricDataList": metrics,
        "BillableTimeInSeconds": response.get("BillableTimeInSeconds"),
        "TrainingTimeInSeconds": response.get("TrainingTimeInSeconds"),
    }


def _processing_outputs(response: Dict[str, Any]) -> Dict[str, Any]:
    outputs = []
    for item in (response.get("ProcessingOutputConfig") or {}).get("Outputs") or []:
        projected = {"OutputName": item.get("OutputName")}
        if item.get("S3Output"):
            projected["S3Uri"] = item["S3Output"].get("S3Uri")
        if item.get("FeatureStoreOutput"):
            projected["FeatureGroupName"] = item["FeatureStoreOutput"].get("FeatureGroupName")
        outputs.append(projected)
    return {
        "ProcessingJobArn": response.get("ProcessingJobArn"),
        "ProcessingJobName": response.get("ProcessingJobName"),
        "Outputs": outputs,
        "ExitMessage": response.get("ExitMessage"),
        "ProcessingStartTime": _iso(response.get("ProcessingStartTime")),
        "ProcessingEndTime": _iso(response.get("ProcessingEndTime")),
    }


def _transform_outputs(response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "TransformJobArn": response.get("TransformJobArn"),
        "TransformJobName": response.get("TransformJobName"),
        "ModelName": response.get("ModelName"),
        "TransformOutput": {
            "S3OutputPath": (response.get("TransformOutput") or {}).get("S3OutputPath"),
        },
        "TransformStartTime": _iso(response.get("TransformStartTime")),
        "TransformEndTime": _iso(response.get("TransformEndTime")),
    }


def _recommendation(item: Dict[str, Any]) -> Dict[str, Any]:
    metrics = item.get("Metrics") or {}
    endpoint = item.get("EndpointConfiguration") or {}
    model = item.get("ModelConfiguration") or {}
    serverless = endpoint.get("ServerlessConfig") or {}
    return {
        "RecommendationId": item.get("RecommendationId"),
        "Metrics": {
            key: metrics.get(key)
            for key in (
                "CostPerHour",
                "CostPerInference",
                "MaxInvocations",
                "ModelLatency",
                "CpuUtilization",
                "MemoryUtilization",
                "ModelSetupTime",
            )
        },
        "EndpointConfiguration": {
            "EndpointName": endpoint.get("EndpointName"),
            "VariantName": endpoint.get("VariantName"),
            "InstanceType": endpoint.get("InstanceType"),
            "InitialInstanceCount": endpoint.get("InitialInstanceCount"),
            "ServerlessConfig": (
                {key: serverless.get(key) for key in ("MemorySizeInMB", "MaxConcurrency", "ProvisionedConcurrency")}
                if serverless
                else None
            ),
        },
        "ModelConfiguration": {
            "InferenceSpecificationName": model.get("InferenceSpecificationName"),
            "CompilationJobName": model.get("CompilationJobName"),
        },
        "InvocationStartTime": _iso(item.get("InvocationStartTime")),
        "InvocationEndTime": _iso(item.get("InvocationEndTime")),
    }


def _inference_outputs(response: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "JobArn": response.get("JobArn"),
        "JobName": response.get("JobName"),
        "JobType": response.get("JobType"),
        "InferenceRecommendations": [_recommendation(item) for item in response.get("InferenceRecommendations") or []],
        "EndpointPerformances": [
            {
                "Metrics": {
                    "MaxInvocations": (item.get("Metrics") or {}).get("MaxInvocations"),
                    "ModelLatency": (item.get("Metrics") or {}).get("ModelLatency"),
                },
                "EndpointInfo": {
                    "EndpointName": (item.get("EndpointInfo") or {}).get("EndpointName"),
                },
            }
            for item in response.get("EndpointPerformances") or []
        ],
        "CompletionTime": _iso(response.get("CompletionTime")),
    }


@dataclass(frozen=True)
class JobDefinition:
    task_type: str
    create_method: str
    describe_method: str
    stop_method: str
    name_key: str
    status_key: str
    state_map: Dict[str, TaskExecution.Phase]
    output_builder: Callable[[Dict[str, Any]], Dict[str, Any]]
    secondary_status_key: Optional[str] = None
    completed_state: str = "Completed"


_PASCAL_STATES = {
    "InProgress": TaskExecution.RUNNING,
    "Stopping": TaskExecution.RUNNING,
    "Completed": TaskExecution.SUCCEEDED,
    "Failed": TaskExecution.FAILED,
    "Stopped": TaskExecution.FAILED,
    "Deleting": TaskExecution.FAILED,
    "DeleteFailed": TaskExecution.FAILED,
}
_INFERENCE_STATES = {
    "PENDING": TaskExecution.RUNNING,
    "IN_PROGRESS": TaskExecution.RUNNING,
    "STOPPING": TaskExecution.RUNNING,
    "COMPLETED": TaskExecution.SUCCEEDED,
    "FAILED": TaskExecution.FAILED,
    "STOPPED": TaskExecution.FAILED,
    "DELETING": TaskExecution.FAILED,
    "DELETED": TaskExecution.FAILED,
}
_ENDPOINT_STATES = {
    "Creating": TaskExecution.RUNNING,
    "Updating": TaskExecution.RUNNING,
    "SystemUpdating": TaskExecution.RUNNING,
    "RollingBack": TaskExecution.RUNNING,
    "InService": TaskExecution.SUCCEEDED,
    "Failed": TaskExecution.FAILED,
    "OutOfService": TaskExecution.FAILED,
    "Deleting": TaskExecution.RUNNING,
}

TRAINING = JobDefinition(
    TRAINING_TASK_TYPE,
    "create_training_job",
    "describe_training_job",
    "stop_training_job",
    "TrainingJobName",
    "TrainingJobStatus",
    _PASCAL_STATES,
    _training_outputs,
    "SecondaryStatus",
)
PROCESSING = JobDefinition(
    PROCESSING_TASK_TYPE,
    "create_processing_job",
    "describe_processing_job",
    "stop_processing_job",
    "ProcessingJobName",
    "ProcessingJobStatus",
    _PASCAL_STATES,
    _processing_outputs,
)
TRANSFORM = JobDefinition(
    TRANSFORM_TASK_TYPE,
    "create_transform_job",
    "describe_transform_job",
    "stop_transform_job",
    "TransformJobName",
    "TransformJobStatus",
    _PASCAL_STATES,
    _transform_outputs,
)
HYPERPARAMETER = JobDefinition(
    HYPERPARAMETER_TUNING_TASK_TYPE,
    "create_hyper_parameter_tuning_job",
    "describe_hyper_parameter_tuning_job",
    "stop_hyper_parameter_tuning_job",
    "HyperParameterTuningJobName",
    "HyperParameterTuningJobStatus",
    _PASCAL_STATES,
    lambda response: response,
)
INFERENCE = JobDefinition(
    INFERENCE_RECOMMENDER_TASK_TYPE,
    "create_inference_recommendations_job",
    "describe_inference_recommendations_job",
    "stop_inference_recommendations_job",
    "JobName",
    "Status",
    _INFERENCE_STATES,
    _inference_outputs,
    completed_state="COMPLETED",
)
ENDPOINT = JobDefinition(
    ENDPOINT_TASK_TYPE,
    "create_endpoint",
    "describe_endpoint",
    "delete_endpoint",
    "EndpointName",
    "EndpointStatus",
    _ENDPOINT_STATES,
    lambda response: {"EndpointArn": response.get("EndpointArn")},
    completed_state="InService",
)


@dataclass
class SageMakerJobMetadata(ResourceMeta):
    config: Dict[str, Any]
    region: str
    inputs: Optional[Dict[str, Any]] = None
    output_key: Optional[str] = "result"

    @property
    def job_name(self) -> str:
        return str(self.config.get("name", ""))


SageMakerTrainingJobMetadata = SageMakerJobMetadata
SageMakerProcessingJobMetadata = SageMakerJobMetadata
SageMakerTransformJobMetadata = SageMakerJobMetadata
SageMakerHyperParameterTuningJobMetadata = SageMakerJobMetadata
SageMakerInferenceRecommenderJobMetadata = SageMakerJobMetadata
SageMakerEndpointMetadata = SageMakerJobMetadata


class SageMakerManagedJobConnector(BotoCaller, AsyncConnector[SageMakerJobMetadata]):
    definition: JobDefinition
    metadata_type = SageMakerJobMetadata

    def _custom(self, task_template: TaskTemplate) -> Dict[str, Any]:
        return normalize_integers(MessageToDict(task_template.custom) if task_template.custom else {})

    def _output_key(self, task_template: TaskTemplate) -> Optional[str]:
        outputs = [item.key for item in task_template.interface.outputs.variables]
        if len(outputs) > 1:
            raise ValueError("SageMaker managed jobs support at most one Flyte output")
        return outputs[0] if outputs else None

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> SageMakerJobMetadata:
        custom = self._custom(task_template)
        config = dict(custom.get("config") or custom.get("request") or {})
        if self.definition.name_key not in config:
            config[self.definition.name_key] = task_template.id.name.rsplit(".", 1)[-1]
        region = resolve_region(custom.get("region"), inputs)
        rendered, _token = render_config(
            config,
            inputs=inputs,
            images=resolve_images(custom.get("images"), region),
            region=region,
            metadata=task_execution_metadata,
        )
        try:
            await self._call(
                self.definition.create_method,
                config,
                inputs=inputs,
                images=custom.get("images"),
                region=region,
                task_execution_metadata=task_execution_metadata,
            )
        except BotoCallError as error:
            if not _duplicate(error):
                raise
        return SageMakerJobMetadata(
            config={"name": rendered[self.definition.name_key]},
            region=region,
            # Inputs are fully rendered into the provider name before metadata is
            # persisted. Avoid retaining arbitrary native values in JSON metadata.
            inputs=None,
            output_key=self._output_key(task_template),
        )

    async def get(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> Resource:
        response, _ = await self._call(
            self.definition.describe_method,
            {self.definition.name_key: resource_meta.config["name"]},
            inputs=resource_meta.inputs,
            region=resource_meta.region,
        )
        state = str(response.get(self.definition.status_key) or "")
        phase = self.definition.state_map.get(state)
        if phase is None:
            return Resource(phase=TaskExecution.FAILED, message=f"Unrecognized SageMaker state {state!r}")
        message = None
        if state in {"InProgress", "PENDING", "IN_PROGRESS"} and self.definition.secondary_status_key:
            message = response.get(self.definition.secondary_status_key)
        elif phase == TaskExecution.FAILED:
            message = response.get("FailureReason") or response.get("ExitMessage")
        outputs = None
        if state == self.definition.completed_state:
            result = await self._build_outputs(response, resource_meta)
            outputs = {resource_meta.output_key: result} if resource_meta.output_key else {}
        return Resource(phase=phase, message=message, outputs=outputs)

    async def _build_outputs(
        self,
        response: Dict[str, Any],
        resource_meta: SageMakerJobMetadata,
    ) -> Dict[str, Any]:
        return self.definition.output_builder(response)

    async def delete(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> None:
        try:
            await self._call(
                self.definition.stop_method,
                {self.definition.name_key: resource_meta.config["name"]},
                inputs=resource_meta.inputs,
                region=resource_meta.region,
            )
        except BotoCallError as error:
            if not _terminal_stop_error(error):
                raise


class SageMakerTrainingJobConnector(SageMakerManagedJobConnector):
    name = "SageMaker Training Job Connector"
    task_type_name = TRAINING_TASK_TYPE
    definition = TRAINING


class SageMakerProcessingJobConnector(SageMakerManagedJobConnector):
    name = "SageMaker Processing Job Connector"
    task_type_name = PROCESSING_TASK_TYPE
    definition = PROCESSING


class SageMakerTransformJobConnector(SageMakerManagedJobConnector):
    name = "SageMaker Transform Job Connector"
    task_type_name = TRANSFORM_TASK_TYPE
    definition = TRANSFORM


class SageMakerHyperParameterTuningJobConnector(SageMakerManagedJobConnector):
    name = "SageMaker Hyperparameter Tuning Job Connector"
    task_type_name = HYPERPARAMETER_TUNING_TASK_TYPE
    definition = HYPERPARAMETER

    async def _build_outputs(
        self,
        response: Dict[str, Any],
        resource_meta: SageMakerJobMetadata,
    ) -> Dict[str, Any]:
        best = response.get("BestTrainingJob") or {}
        artifacts = None
        if best.get("TrainingJobName"):
            training, _ = await self._call(
                "describe_training_job",
                {"TrainingJobName": best["TrainingJobName"]},
                region=resource_meta.region,
                inputs=resource_meta.inputs,
            )
            artifacts = (training.get("ModelArtifacts") or {}).get("S3ModelArtifacts")
        objective = best.get("FinalHyperParameterTuningJobObjectiveMetric") or {}
        counters = response.get("TrainingJobStatusCounters") or {}
        objective_counters = response.get("ObjectiveStatusCounters") or {}
        projected_best = {
            "TrainingJobName": best.get("TrainingJobName"),
            "TrainingJobArn": best.get("TrainingJobArn"),
            "TrainingJobStatus": best.get("TrainingJobStatus"),
            "ObjectiveStatus": best.get("ObjectiveStatus"),
            "FinalHyperParameterTuningJobObjectiveMetric": {
                "MetricName": objective.get("MetricName"),
                "Value": objective.get("Value"),
            },
            "TunedHyperParameters": dict(best.get("TunedHyperParameters") or {}),
            "ModelArtifacts": {"S3ModelArtifacts": artifacts},
            "TrainingStartTime": _iso(best.get("TrainingStartTime")),
            "TrainingEndTime": _iso(best.get("TrainingEndTime")),
        }
        return {
            "HyperParameterTuningJobArn": response.get("HyperParameterTuningJobArn"),
            "HyperParameterTuningJobName": response.get("HyperParameterTuningJobName"),
            "BestTrainingJob": projected_best,
            "ModelArtifacts": {"S3ModelArtifacts": artifacts},
            "TrainingJobStatusCounters": {
                key: counters.get(key)
                for key in ("Completed", "InProgress", "RetryableError", "NonRetryableError", "Stopped")
            },
            "ObjectiveStatusCounters": {key: objective_counters.get(key) for key in ("Succeeded", "Pending", "Failed")},
        }

    async def get(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> Resource:
        resource = await super().get(resource_meta, **kwargs)
        if resource.phase == TaskExecution.RUNNING:
            response, _ = await self._call(
                self.definition.describe_method,
                {self.definition.name_key: resource_meta.config["name"]},
                region=resource_meta.region,
                inputs=resource_meta.inputs,
            )
            counters = response.get("TrainingJobStatusCounters") or {}
            resource.message = (
                f"{counters.get('Completed', 0)} Completed / {counters.get('InProgress', 0)} InProgress / "
                f"{(counters.get('RetryableError') or 0) + (counters.get('NonRetryableError') or 0)} Failed trials"
            )
        return resource


class SageMakerInferenceRecommenderJobConnector(SageMakerManagedJobConnector):
    name = "SageMaker Inference Recommender Job Connector"
    task_type_name = INFERENCE_RECOMMENDER_TASK_TYPE
    definition = INFERENCE


class SageMakerEndpointConnector(SageMakerManagedJobConnector):
    name = "SageMaker Endpoint Connector"
    task_type_name = ENDPOINT_TASK_TYPE
    definition = ENDPOINT


# Short names retained from the initial Flyte 2 draft.
SageMakerTrainingConnector = SageMakerTrainingJobConnector
SageMakerProcessingConnector = SageMakerProcessingJobConnector
SageMakerTransformConnector = SageMakerTransformJobConnector


@dataclass
class BotoCallMetadata(ResourceMeta):
    response: Dict[str, Any]
    idempotence_token: str
    output_names: list[str]


class BotoConnector(BotoCaller, AsyncConnector[BotoCallMetadata]):
    """Invoke one public boto3 client method."""

    name = "Boto Connector"
    task_type_name = BOTO_TASK_TYPE
    metadata_type = BotoCallMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> BotoCallMetadata:
        custom = normalize_integers(MessageToDict(task_template.custom) if task_template.custom else {})
        service = custom.get("service") or ""
        method = custom.get("method") or ""
        if not _SERVICE_NAME.fullmatch(service):
            raise ValueError(f"Invalid boto3 service name {service!r}")
        if not _METHOD_NAME.fullmatch(method):
            raise ValueError(f"Invalid boto3 method name {method!r}")
        try:
            response, token = await self._call(
                method,
                custom.get("config") or custom.get("request") or {},
                service=service,
                region=custom.get("region"),
                images=custom.get("images"),
                inputs=inputs,
                task_execution_metadata=task_execution_metadata,
            )
        except BotoCallError as error:
            code, message = _error_parts(error)
            if code != "ValidationException" or "Cannot create already existing" not in message:
                raise
            match = re.search(r"arn:aws(?:-[a-z]+)?:[A-Za-z0-9-]+:[A-Za-z0-9-]+:\d+:[A-Za-z0-9/_-]+", message)
            if method == "create_model" and match:
                response = {"ModelArn": match.group(0)}
            elif method == "create_endpoint_config" and match:
                response = {"EndpointConfigArn": match.group(0)}
            else:
                response = {"result": f"Entity already exists{f' {match.group(0)}' if match else ''}."}
            token = error.idempotence_token
        outputs = [item.key for item in task_template.interface.outputs.variables]
        return BotoCallMetadata(response=response, idempotence_token=token, output_names=outputs)

    async def get(self, resource_meta: BotoCallMetadata, **kwargs: Any) -> Resource:
        outputs: Dict[str, Any] = {}
        if "result" in resource_meta.output_names:
            outputs["result"] = resource_meta.response or {"result": None}
        if "idempotence_token" in resource_meta.output_names:
            outputs["idempotence_token"] = resource_meta.idempotence_token
        if len(resource_meta.output_names) == 1 and not outputs:
            outputs[resource_meta.output_names[0]] = resource_meta.response
        return Resource(phase=TaskExecution.SUCCEEDED, outputs=outputs)

    async def delete(self, resource_meta: BotoCallMetadata, **kwargs: Any) -> None:
        return None


@dataclass
class PythonicJobMetadata(ResourceMeta):
    job_name: str
    output_prefix: str
    region: str
    has_outputs: bool


def _job_name(
    prefix: str,
    metadata: Optional[TaskExecutionMetadata],
    task_template: TaskTemplate,
    output_prefix: str,
) -> str:
    parts = [output_prefix, task_template.id.name, task_template.id.version]
    if metadata is not None and metadata.HasField("task_execution_id"):
        parts.append(metadata.task_execution_id.SerializeToString().hex())
    digest = hashlib.sha256("-".join(parts).encode()).hexdigest()[:20]
    prefix = re.sub(r"[^A-Za-z0-9-]", "-", prefix).strip("-") or "flyte"
    return f"{prefix[:42].rstrip('-')}-{digest}"


def _container_entrypoint(task_template: TaskTemplate) -> list[str]:
    if task_template.container is None or not task_template.container.image:
        raise ValueError("Pythonic SageMaker jobs require a container image")
    entrypoint = [*task_template.container.command, *task_template.container.args]
    if not entrypoint or "a0" not in entrypoint:
        raise ValueError("Pythonic SageMaker jobs require the rendered Flyte a0 entrypoint")
    if len(entrypoint) > 100:
        raise ValueError("SageMaker ContainerEntrypoint supports at most 100 arguments")
    if any(len(item) > 256 for item in entrypoint):
        raise ValueError("Each SageMaker ContainerEntrypoint argument must be at most 256 characters")
    return entrypoint


def _execution_environment(
    task_template: TaskTemplate,
    metadata: Optional[TaskExecutionMetadata],
    configured: Optional[Dict[str, str]],
) -> Dict[str, str]:
    environment = {item.key: item.value for item in task_template.container.env}
    if metadata is not None:
        environment.update(metadata.environment_variables)
        if metadata.HasField("task_execution_id"):
            task_id = metadata.task_execution_id
            execution = task_id.node_execution_id.execution_id
            environment.update(
                {
                    "FLYTE_INTERNAL_EXECUTION_PROJECT": execution.project,
                    "FLYTE_INTERNAL_EXECUTION_DOMAIN": execution.domain,
                    "FLYTE_INTERNAL_EXECUTION_ID": execution.name,
                    "FLYTE_ATTEMPT_NUMBER": str(task_id.retry_attempt),
                }
            )
    environment.update(configured or {})
    environment["FLYTE_FAIL_ON_ERROR"] = "true"
    return {key: value for key, value in environment.items() if value != ""}


class PythonicSageMakerJobConnector(BotoCaller, AsyncConnector[PythonicJobMetadata]):
    metadata_type = PythonicJobMetadata
    create_method: str
    describe_method: str
    stop_method: str
    name_key: str
    status_key: str
    secondary_status_key: Optional[str] = None
    state_map = _PASCAL_STATES

    def _build_request(
        self,
        task_template: TaskTemplate,
        custom: Dict[str, Any],
        job_name: str,
        output_prefix: str,
        environment: Dict[str, str],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> PythonicJobMetadata:
        custom = normalize_integers(MessageToDict(task_template.custom) if task_template.custom else {})
        region = resolve_region(custom.get("region"), inputs)
        name = _job_name(
            custom.get("job_name_prefix") or "flyte-",
            task_execution_metadata,
            task_template,
            output_prefix,
        )
        environment = _execution_environment(task_template, task_execution_metadata, custom.get("environment"))
        request = self._build_request(task_template, custom, name, output_prefix, environment)
        try:
            await self._call(self.create_method, request, region=region)
        except BotoCallError as error:
            code, message = _error_parts(error)
            if code not in _DUPLICATE_CODES and not (
                code == "ValidationException" and "Cannot create already existing" in message
            ):
                raise
        return PythonicJobMetadata(
            job_name=name,
            output_prefix=output_prefix,
            region=region,
            has_outputs=bool(task_template.interface.outputs.variables),
        )

    async def get(self, resource_meta: PythonicJobMetadata, **kwargs: Any) -> Resource:
        response, _ = await self._call(
            self.describe_method,
            {self.name_key: resource_meta.job_name},
            region=resource_meta.region,
        )
        state = str(response.get(self.status_key) or "")
        phase = self.state_map.get(state, TaskExecution.RUNNING)
        message = None
        if state == "InProgress" and self.secondary_status_key:
            message = response.get(self.secondary_status_key)
        elif state in {"Failed", "Stopped"}:
            message = response.get("FailureReason") or response.get("ExitMessage")
        if state in {"Completed", "Failed", "Stopped"}:
            error = await self._read_error(resource_meta.output_prefix)
            if error is not None:
                return Resource(
                    phase=TaskExecution.RETRYABLE_FAILED if error[1] else TaskExecution.FAILED,
                    message=error[0],
                )
        if state == "Completed":
            if resource_meta.has_outputs and not await storage.exists(
                f"{resource_meta.output_prefix.rstrip('/')}/outputs.pb"
            ):
                return Resource(
                    phase=TaskExecution.FAILED,
                    message="SageMaker job completed without producing Flyte outputs.pb.",
                )
            return Resource(phase=TaskExecution.SUCCEEDED, outputs=None, message=message)
        return Resource(phase=phase, message=message)

    async def delete(self, resource_meta: PythonicJobMetadata, **kwargs: Any) -> None:
        try:
            await self._call(
                self.stop_method,
                {self.name_key: resource_meta.job_name},
                region=resource_meta.region,
            )
        except BotoCallError as error:
            if not _terminal_stop_error(error):
                raise

    @staticmethod
    async def _read_error(output_prefix: str) -> Optional[tuple[str, bool]]:
        path = f"{output_prefix.rstrip('/')}/error.pb"
        if not await storage.exists(path):
            return None
        try:
            payload = b"".join([chunk async for chunk in storage.get_stream(path)])
            document = execution_pb2.ErrorDocument()
            document.ParseFromString(payload)
            return (
                document.error.message or "User task failed.",
                document.error.kind == execution_pb2.ContainerError.RECOVERABLE,
            )
        except Exception:
            return ("User task raised an error, but error.pb could not be decoded.", False)


def _tags(tags: Optional[Dict[str, str]]) -> Optional[list[dict[str, str]]]:
    return [{"Key": str(key), "Value": str(value)} for key, value in tags.items()] if tags else None


class SageMakerTrainingTaskConnector(PythonicSageMakerJobConnector):
    name = "SageMaker Training Task Connector"
    task_type_name = TRAINING_PYTHON_TASK_TYPE
    create_method = "create_training_job"
    describe_method = "describe_training_job"
    stop_method = "stop_training_job"
    name_key = "TrainingJobName"
    status_key = "TrainingJobStatus"
    secondary_status_key = "SecondaryStatus"

    def _build_request(
        self,
        task_template: TaskTemplate,
        custom: Dict[str, Any],
        job_name: str,
        output_prefix: str,
        environment: Dict[str, str],
    ) -> Dict[str, Any]:
        resource = {
            "InstanceType": custom.get("instance_type", "ml.m5.large"),
            "InstanceCount": custom.get("instance_count", 1),
            "VolumeSizeInGB": custom.get("volume_size_in_gb", 30),
        }
        if custom.get("kms_key_id"):
            resource["VolumeKmsKeyId"] = custom["kms_key_id"]
        request = {
            "TrainingJobName": job_name,
            "RoleArn": custom["execution_role_arn"],
            "AlgorithmSpecification": {
                "TrainingImage": task_template.container.image,
                "ContainerEntrypoint": _container_entrypoint(task_template),
                "TrainingInputMode": "File",
            },
            "ResourceConfig": resource,
            "OutputDataConfig": {
                "S3OutputPath": custom.get("output_s3_path") or f"{output_prefix.rstrip('/')}/_sagemaker_model"
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": custom.get("max_runtime_in_seconds", 3600)},
            "Environment": environment,
        }
        if custom.get("vpc_config"):
            request["VpcConfig"] = custom["vpc_config"]
        if _tags(custom.get("tags")):
            request["Tags"] = _tags(custom["tags"])
        return request


class SageMakerProcessingTaskConnector(PythonicSageMakerJobConnector):
    name = "SageMaker Processing Task Connector"
    task_type_name = PROCESSING_PYTHON_TASK_TYPE
    create_method = "create_processing_job"
    describe_method = "describe_processing_job"
    stop_method = "stop_processing_job"
    name_key = "ProcessingJobName"
    status_key = "ProcessingJobStatus"

    def _build_request(
        self,
        task_template: TaskTemplate,
        custom: Dict[str, Any],
        job_name: str,
        output_prefix: str,
        environment: Dict[str, str],
    ) -> Dict[str, Any]:
        cluster = {
            "InstanceType": custom.get("instance_type", "ml.m5.large"),
            "InstanceCount": custom.get("instance_count", 1),
            "VolumeSizeInGB": custom.get("volume_size_in_gb", 30),
        }
        if custom.get("kms_key_id"):
            cluster["VolumeKmsKeyId"] = custom["kms_key_id"]
        network = custom.get("network_config")
        if network and network.get("EnableNetworkIsolation"):
            raise ValueError(
                "EnableNetworkIsolation=True is incompatible with Pythonic mode because a0 must access Flyte storage."
            )
        request = {
            "ProcessingJobName": job_name,
            "RoleArn": custom["execution_role_arn"],
            "AppSpecification": {
                "ImageUri": task_template.container.image,
                "ContainerEntrypoint": _container_entrypoint(task_template),
            },
            "ProcessingResources": {"ClusterConfig": cluster},
            "StoppingCondition": {"MaxRuntimeInSeconds": custom.get("max_runtime_in_seconds", 3600)},
            "Environment": environment,
        }
        if network:
            request["NetworkConfig"] = network
        if _tags(custom.get("tags")):
            request["Tags"] = _tags(custom["tags"])
        return request


for connector in (
    SageMakerTrainingJobConnector(),
    SageMakerProcessingJobConnector(),
    SageMakerTransformJobConnector(),
    SageMakerHyperParameterTuningJobConnector(),
    SageMakerInferenceRecommenderJobConnector(),
    SageMakerEndpointConnector(),
    SageMakerTrainingTaskConnector(),
    SageMakerProcessingTaskConnector(),
    BotoConnector(),
):
    ConnectorRegistry.register(connector)

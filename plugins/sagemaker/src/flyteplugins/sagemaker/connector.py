from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from flyteplugins.sagemaker.task import (
    PROCESSING_TASK_TYPE,
    TRAINING_TASK_TYPE,
    TRANSFORM_TASK_TYPE,
)

_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "adaptive"})
_RUNNING_STATES = {"Starting", "InProgress", "Stopping"}
_RETRYABLE_ERROR_CODES = {
    "InternalFailure",
    "InternalServerException",
    "ServiceUnavailable",
    "ThrottlingException",
    "TooManyRequestsException",
}


def _restore_integer_values(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_restore_integer_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_integer_values(item) for key, item in value.items()}
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class _JobDefinition:
    task_type: str
    create_method: str
    describe_method: str
    stop_method: str
    name_key: str
    status_key: str
    failure_key: str
    console_path: str
    result_fields: Dict[str, str]


_TRAINING = _JobDefinition(
    task_type=TRAINING_TASK_TYPE,
    create_method="create_training_job",
    describe_method="describe_training_job",
    stop_method="stop_training_job",
    name_key="TrainingJobName",
    status_key="TrainingJobStatus",
    failure_key="FailureReason",
    console_path="jobs",
    result_fields={
        "ModelArtifacts": "model_artifacts",
        "OutputDataConfig": "output_data_config",
        "TrainingTimeInSeconds": "training_time_seconds",
        "BillableTimeInSeconds": "billable_time_seconds",
    },
)
_PROCESSING = _JobDefinition(
    task_type=PROCESSING_TASK_TYPE,
    create_method="create_processing_job",
    describe_method="describe_processing_job",
    stop_method="stop_processing_job",
    name_key="ProcessingJobName",
    status_key="ProcessingJobStatus",
    failure_key="FailureReason",
    console_path="processing-jobs",
    result_fields={
        "ProcessingOutputConfig": "processing_output_config",
        "ProcessingResources": "processing_resources",
    },
)
_TRANSFORM = _JobDefinition(
    task_type=TRANSFORM_TASK_TYPE,
    create_method="create_transform_job",
    describe_method="describe_transform_job",
    stop_method="stop_transform_job",
    name_key="TransformJobName",
    status_key="TransformJobStatus",
    failure_key="FailureReason",
    console_path="transform-jobs",
    result_fields={
        "TransformOutput": "transform_output",
        "TransformResources": "transform_resources",
    },
)


@dataclass
class SageMakerJobMetadata(ResourceMeta):
    region: Optional[str]
    job_name: str
    output_key: Optional[str] = None


class SageMakerManagedJobConnector(AsyncConnector[SageMakerJobMetadata]):
    definition: _JobDefinition
    metadata_type = SageMakerJobMetadata

    def _get_client(self, region: Optional[str]) -> Any:
        kwargs: Dict[str, Any] = {
            "service_name": "sagemaker",
            "config": _BOTO_CONFIG,
        }
        if region:
            kwargs["region_name"] = region
        return boto3.client(**kwargs)

    async def _call(self, client: Any, method: str, **request: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(getattr(client, method), **request)

    def _output_key(self, task_template: TaskTemplate) -> Optional[str]:
        outputs = [variable.key for variable in task_template.interface.outputs.variables]
        if len(outputs) > 1:
            raise ValueError("SageMaker managed jobs support at most one Flyte output")
        return outputs[0] if outputs else None

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SageMakerJobMetadata:
        custom = MessageToDict(task_template.custom) if task_template.custom else {}
        request = _restore_integer_values(custom.get("request") or {})
        region = custom.get("region") or None
        job_name = request.get(self.definition.name_key)
        if not job_name:
            raise ValueError(f"SageMaker request.{self.definition.name_key} is required")

        client = self._get_client(region)
        try:
            await self._call(client, self.definition.create_method, **request)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code not in {"ConflictException", "ResourceInUse"}:
                raise
            await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: job_name},
            )

        return SageMakerJobMetadata(
            region=region,
            job_name=job_name,
            output_key=self._output_key(task_template),
        )

    async def get(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> Resource:
        client = self._get_client(resource_meta.region)
        try:
            response = await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: resource_meta.job_name},
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            phase = TaskExecution.RETRYABLE_FAILED if code in _RETRYABLE_ERROR_CODES else TaskExecution.FAILED
            return Resource(phase=phase, message=str(error))

        state = response.get(self.definition.status_key, "")
        if state in _RUNNING_STATES:
            phase = TaskExecution.RUNNING
        elif state == "Completed":
            phase = TaskExecution.SUCCEEDED
        elif state == "Stopped":
            phase = TaskExecution.ABORTED
        elif state == "Failed":
            phase = TaskExecution.FAILED
        else:
            return Resource(
                phase=TaskExecution.FAILED,
                message=f"Unrecognized SageMaker job state {state!r}",
            )

        region = resource_meta.region or getattr(client.meta, "region_name", None) or "us-east-1"
        job_type = self.definition.task_type.removeprefix("sagemaker-").removesuffix("-job").title()
        log = TaskLog(
            uri=(
                f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}"
                f"#/{self.definition.console_path}/{resource_meta.job_name}"
            ),
            name=f"SageMaker {job_type} Console",
            ready=True,
            link_type=TaskLog.EXTERNAL,
        )
        message = state
        failure = response.get(self.definition.failure_key)
        if failure:
            message = f"{state}: {failure}"

        outputs: Optional[Dict[str, Any]] = None
        if phase == TaskExecution.SUCCEEDED:
            result: Dict[str, Any] = {
                "job_name": resource_meta.job_name,
                "status": state,
            }
            for aws_key, output_key in self.definition.result_fields.items():
                if aws_key in response:
                    result[output_key] = _json_safe(response[aws_key])
            outputs = {resource_meta.output_key: result} if resource_meta.output_key else {}

        return Resource(
            phase=phase,
            message=message,
            log_links=[log],
            outputs=outputs,
            custom_info={"job_name": resource_meta.job_name},
        )

    async def delete(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> None:
        client = self._get_client(resource_meta.region)
        try:
            response = await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: resource_meta.job_name},
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in {"ResourceNotFound", "ValidationException"}:
                return
            raise
        if response.get(self.definition.status_key) in {"Completed", "Failed", "Stopped"}:
            return
        await self._call(
            client,
            self.definition.stop_method,
            **{self.definition.name_key: resource_meta.job_name},
        )


class SageMakerTrainingConnector(SageMakerManagedJobConnector):
    name = "SageMaker Training Connector"
    task_type_name = TRAINING_TASK_TYPE
    definition = _TRAINING


class SageMakerProcessingConnector(SageMakerManagedJobConnector):
    name = "SageMaker Processing Connector"
    task_type_name = PROCESSING_TASK_TYPE
    definition = _PROCESSING


class SageMakerTransformConnector(SageMakerManagedJobConnector):
    name = "SageMaker Transform Connector"
    task_type_name = TRANSFORM_TASK_TYPE
    definition = _TRANSFORM


ConnectorRegistry.register(SageMakerTrainingConnector())
ConnectorRegistry.register(SageMakerProcessingConnector())
ConnectorRegistry.register(SageMakerTransformConnector())

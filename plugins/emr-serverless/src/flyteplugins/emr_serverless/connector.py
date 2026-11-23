from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flyte import system_logger as logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from flyteplugins.emr_serverless.task import TASK_TYPE

_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "adaptive"})
_APPLICATION_READY = "STARTED"
_APPLICATION_STARTABLE = {"CREATED", "STOPPED"}
_APPLICATION_TRANSITIONING = {"CREATING", "STARTING", "STOPPING"}
_APPLICATION_TERMINAL = {"TERMINATED"}
_JOB_RUNNING = {"PENDING", "SCHEDULED", "SUBMITTED", "RUNNING", "CANCELLING"}
_JOB_TERMINAL = {"SUCCESS", "FAILED", "CANCELLED"}
_RETRYABLE_ERROR_CODES = {
    "InternalServerException",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
}


def _restore_integer_values(value: Any) -> Any:
    """Restore whole numbers converted to floats by protobuf `Struct`."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_restore_integer_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_integer_values(item) for key, item in value.items()}
    return value


def _execution_token(metadata: Optional[TaskExecutionMetadata], request: Dict[str, Any]) -> str:
    if metadata is not None and metadata.HasField("task_execution_id"):
        source = metadata.task_execution_id.SerializeToString()
    else:
        source = json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    return hashlib.sha256(source).hexdigest()


@dataclass
class EMRServerlessMetadata(ResourceMeta):
    application_id: str
    region: Optional[str]
    job_run_id: str = ""
    pending_request: Optional[Dict[str, Any]] = None


class EMRServerlessConnector(AsyncConnector[EMRServerlessMetadata]):
    """Submit and monitor existing Spark or Hive jobs on EMR Serverless."""

    name = "EMR Serverless Connector"
    task_type_name = TASK_TYPE
    metadata_type = EMRServerlessMetadata

    def _get_client(self, region: Optional[str]) -> Any:
        kwargs: Dict[str, Any] = {
            "service_name": "emr-serverless",
            "config": _BOTO_CONFIG,
        }
        if region:
            kwargs["region_name"] = region
        return boto3.client(**kwargs)

    async def _call(self, client: Any, method: str, **request: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(getattr(client, method), **request)

    async def _start_application_if_needed(self, client: Any, application_id: str) -> bool:
        response = await self._call(client, "get_application", applicationId=application_id)
        state = response.get("application", {}).get("state", "")
        if state == _APPLICATION_READY:
            return True
        if state in _APPLICATION_TERMINAL:
            raise RuntimeError(f"EMR Serverless application {application_id} is in terminal state {state!r}")
        if state in _APPLICATION_STARTABLE:
            await self._call(client, "start_application", applicationId=application_id)
            return False
        if state in _APPLICATION_TRANSITIONING:
            return False
        raise RuntimeError(f"Unrecognized EMR Serverless application state {state!r}")

    async def _submit(self, client: Any, request: Dict[str, Any]) -> str:
        response = await self._call(client, "start_job_run", **request)
        job_run_id = response.get("jobRunId")
        if not job_run_id:
            raise RuntimeError("EMR Serverless StartJobRun response did not include jobRunId")
        return str(job_run_id)

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> EMRServerlessMetadata:
        custom = MessageToDict(task_template.custom) if task_template.custom else {}
        request = _restore_integer_values(custom.get("request") or {})
        region = custom.get("region") or None
        application_id = request.get("applicationId")
        if not application_id:
            raise ValueError("EMR Serverless request.applicationId is required")
        if not request.get("executionRoleArn"):
            raise ValueError("EMR Serverless request.executionRoleArn is required")
        if not request.get("jobDriver"):
            raise ValueError("EMR Serverless request.jobDriver is required")

        request.setdefault("clientToken", _execution_token(task_execution_metadata, request))
        client = self._get_client(region)
        if not await self._start_application_if_needed(client, application_id):
            return EMRServerlessMetadata(
                application_id=application_id,
                region=region,
                pending_request=request,
            )

        job_run_id = await self._submit(client, request)
        return EMRServerlessMetadata(
            application_id=application_id,
            region=region,
            job_run_id=job_run_id,
        )

    async def get(self, resource_meta: EMRServerlessMetadata, **kwargs: Any) -> Resource:
        client = self._get_client(resource_meta.region)
        try:
            job_run_id = resource_meta.job_run_id
            if not job_run_id:
                if not resource_meta.pending_request:
                    return Resource(
                        phase=TaskExecution.FAILED,
                        message="EMR Serverless job submission metadata is missing",
                    )
                if not await self._start_application_if_needed(client, resource_meta.application_id):
                    return Resource(
                        phase=TaskExecution.RUNNING,
                        message=f"EMR Serverless application {resource_meta.application_id} is starting",
                    )
                job_run_id = await self._submit(client, resource_meta.pending_request)

            response = await self._call(
                client,
                "get_job_run",
                applicationId=resource_meta.application_id,
                jobRunId=job_run_id,
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            phase = TaskExecution.RETRYABLE_FAILED if code in _RETRYABLE_ERROR_CODES else TaskExecution.FAILED
            return Resource(phase=phase, message=str(error))

        job = response.get("jobRun", {})
        state = job.get("state", "")
        details = job.get("stateDetails") or ""
        if state in _JOB_RUNNING:
            phase = TaskExecution.RUNNING
        elif state == "SUCCESS":
            phase = TaskExecution.SUCCEEDED
        elif state == "CANCELLED":
            phase = TaskExecution.ABORTED
        elif state == "FAILED":
            phase = TaskExecution.FAILED
        else:
            return Resource(
                phase=TaskExecution.FAILED,
                message=f"Unrecognized EMR Serverless job state {state!r}",
            )

        region = resource_meta.region or getattr(client.meta, "region_name", None) or "us-east-1"
        log = TaskLog(
            uri=(
                f"https://{region}.console.aws.amazon.com/emr/home?region={region}"
                f"#/serverless/{resource_meta.application_id}/jobs/{job_run_id}"
            ),
            name="EMR Serverless Console",
            ready=True,
            link_type=TaskLog.EXTERNAL,
        )
        message = f"EMR Serverless job state: {state}"
        if details:
            message = f"{message} - {details}"
        return Resource(
            phase=phase,
            message=message,
            log_links=[log],
            outputs={} if phase == TaskExecution.SUCCEEDED else None,
            custom_info={
                "application_id": resource_meta.application_id,
                "job_run_id": job_run_id,
            },
        )

    async def delete(self, resource_meta: EMRServerlessMetadata, **kwargs: Any) -> None:
        client = self._get_client(resource_meta.region)
        job_run_id = resource_meta.job_run_id
        if not job_run_id and resource_meta.pending_request:
            if not await self._start_application_if_needed(client, resource_meta.application_id):
                return
            job_run_id = await self._submit(client, resource_meta.pending_request)
        if not job_run_id:
            return

        try:
            response = await self._call(
                client,
                "get_job_run",
                applicationId=resource_meta.application_id,
                jobRunId=job_run_id,
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException":
                return
            raise
        state = response.get("jobRun", {}).get("state", "")
        if state in _JOB_TERMINAL:
            return
        try:
            await self._call(
                client,
                "cancel_job_run",
                applicationId=resource_meta.application_id,
                jobRunId=job_run_id,
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException" or (
                code == "ValidationException" and "cannot be cancelled" in str(error).lower()
            ):
                return
            raise
        logger.info("Cancelled EMR Serverless job %s", job_run_id)


ConnectorRegistry.register(EMRServerlessConnector())

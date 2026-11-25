from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from flyte import system_logger as logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from flyteplugins.emr_serverless.boto_handler import EMRServerlessHandler
from flyteplugins.emr_serverless.task import EMR_SERVERLESS_BASE_IMAGE, TASK_TYPE, EMRServerless

_JOB_RUNNING = {
    "QUEUED",
    "PENDING",
    "SCHEDULED",
    "SUBMITTED",
    "RUNNING",
    "CANCELLING",
}
_RETRYABLE_ERROR_CODES = {
    "InternalServerException",
    "ServiceUnavailableException",
    "ThrottlingException",
    "TooManyRequestsException",
}
FLYTE_TAGS = {"Application": "flyte", "ManagedBy": "flyte-connector"}
_APPLICATION_ID_RE = re.compile(r"^[0-9a-z]+$")
_UNSAFE_ENV_VALUE_CHARS = re.compile(r"[\s='\"]")

_ENV_ALLOW_CREATE_APPLICATION = "FLYTE_EMR_ALLOW_CREATE_APPLICATION"
_ENV_APPLICATION_NAME_PREFIX = "FLYTE_EMR_APPLICATION_NAME_PREFIX"
_ENV_ENTRYPOINT_BUCKET = "FLYTE_EMR_ENTRYPOINT_S3_BUCKET"
_ENV_FALLBACK_BUCKET = "FLYTE_AWS_S3_BUCKET"
_ENV_ALLOWED_REGIONS = "FLYTE_EMR_ALLOWED_REGIONS"
_ENV_ALLOWED_EXECUTION_ROLE_ARNS = "FLYTE_EMR_ALLOWED_EXECUTION_ROLE_ARNS"
_ENV_ALLOWED_APPLICATION_IDS = "FLYTE_EMR_ALLOWED_APPLICATION_IDS"
_ENV_ALLOWED_APPLICATION_NAME_PREFIXES = "FLYTE_EMR_ALLOWED_APPLICATION_NAME_PREFIXES"


def _policy_values(name: str) -> frozenset[str]:
    return frozenset(value.strip() for value in os.getenv(name, "").split(",") if value.strip())


@dataclass(frozen=True)
class EMRServerlessConnectorPolicy:
    """Operator-controlled boundary for task-supplied AWS resources."""

    allowed_regions: frozenset[str]
    allowed_execution_role_arns: frozenset[str]
    allowed_application_ids: frozenset[str]
    allowed_application_name_prefixes: frozenset[str]

    @classmethod
    def from_env(cls) -> EMRServerlessConnectorPolicy:
        """Load exact allowlists from the connector process environment."""
        return cls(
            allowed_regions=_policy_values(_ENV_ALLOWED_REGIONS),
            allowed_execution_role_arns=_policy_values(_ENV_ALLOWED_EXECUTION_ROLE_ARNS),
            allowed_application_ids=_policy_values(_ENV_ALLOWED_APPLICATION_IDS),
            allowed_application_name_prefixes=_policy_values(_ENV_ALLOWED_APPLICATION_NAME_PREFIXES),
        )

    @staticmethod
    def _require_allowed(value: str, allowed: frozenset[str], env_name: str, resource: str) -> None:
        if not allowed:
            raise ValueError(f"The connector operator must configure {env_name} before accepting {resource}")
        if value not in allowed:
            raise ValueError(f"{resource} {value!r} is not allowed by {env_name}")

    def validate_region(self, region: str) -> None:
        """Reject task-selected regions outside the operator policy."""
        self._require_allowed(region, self.allowed_regions, _ENV_ALLOWED_REGIONS, "AWS region")

    def validate_execution_role(self, role_arn: str) -> None:
        """Reject task-selected execution roles outside the operator policy."""
        self._require_allowed(
            role_arn,
            self.allowed_execution_role_arns,
            _ENV_ALLOWED_EXECUTION_ROLE_ARNS,
            "execution role",
        )

    def authorizes_application(self, application_id: str, application: Dict[str, Any]) -> bool:
        """Return whether an existing application is explicitly allowed or connector-owned."""
        if application_id in self.allowed_application_ids:
            return True
        tags = application.get("tags") or {}
        name = str(application.get("name") or "")
        return all(tags.get(key) == value for key, value in FLYTE_TAGS.items()) and any(
            name.startswith(prefix) for prefix in self.allowed_application_name_prefixes
        )


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
class EMRServerlessJobMetadata(ResourceMeta):
    """State persisted between connector calls."""

    application_id: str
    job_run_id: str = ""
    region: Optional[str] = None
    created_application: bool = False
    is_script_mode: bool = False
    pending_job_request: Optional[Dict[str, Any]] = None
    pending_image_configuration: Optional[Dict[str, Any]] = None


class EMRServerlessConnector(AsyncConnector[EMRServerlessJobMetadata]):
    """Submit and monitor script and Pythonic jobs on EMR Serverless."""

    name = "EMR Serverless Connector"
    task_type_name = TASK_TYPE
    metadata_type = EMRServerlessJobMetadata
    _ENTRYPOINT_PATH = (Path(__file__).parent / "_entrypoint.py").resolve()
    _entrypoint_bytes_cache: Optional[bytes] = None
    _PYTHONIC_SPARK_DEFAULTS = (
        "--conf spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON=/usr/bin/python3.11 "
        "--conf spark.emr-serverless.driverEnv.PYSPARK_PYTHON=/usr/bin/python3.11 "
        "--conf spark.executorEnv.PYSPARK_PYTHON=/usr/bin/python3.11"
    )

    def __init__(self, policy: Optional[EMRServerlessConnectorPolicy] = None) -> None:
        super().__init__()
        self._policy = policy

    def _get_handler(self, region: Optional[str] = None) -> EMRServerlessHandler:
        return EMRServerlessHandler(region)

    def _get_policy(self) -> EMRServerlessConnectorPolicy:
        return self._policy or EMRServerlessConnectorPolicy.from_env()

    @staticmethod
    def _extract_config(task_template: TaskTemplate) -> EMRServerless:
        custom = _restore_integer_values(MessageToDict(task_template.custom) if task_template.custom else {})
        if not custom:
            raise ValueError("Task template has no EMR Serverless configuration")
        return EMRServerless.from_dict(custom)

    @classmethod
    def _read_entrypoint_bytes(cls) -> bytes:
        """Return the exact bootstrap bytes staged for Pythonic jobs."""
        if cls._entrypoint_bytes_cache is None:
            cls._entrypoint_bytes_cache = cls._ENTRYPOINT_PATH.read_bytes()
        return cls._entrypoint_bytes_cache

    @staticmethod
    def _resolve_entrypoint_bucket(config: EMRServerless) -> str:
        """Resolve the S3 bucket used for the Pythonic bootstrap."""
        configured = os.getenv(_ENV_ENTRYPOINT_BUCKET)
        if configured:
            return configured.removeprefix("s3://").split("/", 1)[0]
        if config.configuration_overrides:
            log_uri = (
                config.configuration_overrides.get("monitoringConfiguration", {})
                .get("s3MonitoringConfiguration", {})
                .get("logUri", "")
            )
            if log_uri.startswith("s3://"):
                return log_uri.removeprefix("s3://").split("/", 1)[0]
        configured = os.getenv(_ENV_FALLBACK_BUCKET)
        if configured:
            return configured.removeprefix("s3://").split("/", 1)[0]
        raise ValueError(
            "Pythonic mode requires an S3 staging bucket. Set "
            f"{_ENV_ENTRYPOINT_BUCKET}, configure an S3 monitoring log URI, or set {_ENV_FALLBACK_BUCKET}."
        )

    def _ensure_entrypoint_on_s3(self, handler: EMRServerlessHandler, config: EMRServerless) -> str:
        """Stage the content-addressed Pythonic bootstrap and return its URI."""
        content = self._read_entrypoint_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        bucket = self._resolve_entrypoint_bucket(config)
        key = f"flyte/emr-serverless/entrypoint-{digest}.py"
        client = boto3.client("s3", region_name=handler.region)
        try:
            client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                raise
            client.put_object(Bucket=bucket, Key=key, Body=content, ContentType="text/x-python")
        return f"s3://{bucket}/{key}"

    @staticmethod
    def _merge_tags(tags: Optional[Dict[str, str]]) -> Dict[str, str]:
        result = dict(tags or {})
        result.update(FLYTE_TAGS)
        return result

    @staticmethod
    def _is_create_application_allowed() -> bool:
        return os.getenv(_ENV_ALLOW_CREATE_APPLICATION, "false").strip().lower() == "true"

    @staticmethod
    def _apply_application_name_prefix(name: str) -> str:
        prefix = os.getenv(_ENV_APPLICATION_NAME_PREFIX, "").strip()
        return f"{prefix}{name}" if prefix and not name.startswith(prefix) else name

    @staticmethod
    def _container_image(task_template: TaskTemplate) -> Optional[str]:
        container = task_template.container
        return container.image if container and container.image else None

    @classmethod
    def _resolve_image_configuration(
        cls,
        task_template: TaskTemplate,
        config: EMRServerless,
        *,
        for_create: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if config.image_configuration:
            return dict(config.image_configuration)
        if config.is_script_mode:
            return None
        image = cls._container_image(task_template)
        if image:
            return {"imageUri": image}
        if for_create:
            return {"imageUri": EMR_SERVERLESS_BASE_IMAGE}
        raise ValueError(
            "Pythonic mode requires a worker image containing Flyte SDK. Set the task image or image_configuration."
        )

    async def _sync_application_image(
        self,
        handler: EMRServerlessHandler,
        application_id: str,
        task_template: TaskTemplate,
        config: EMRServerless,
    ) -> Optional[Dict[str, Any]]:
        if config.is_script_mode and not config.image_configuration:
            return None
        desired_uri = self._container_image(task_template)
        if not desired_uri and config.image_configuration:
            desired_uri = config.image_configuration.get("imageUri")
        if not desired_uri:
            return None
        desired = {"imageUri": desired_uri}
        current = (await handler.get_application(application_id)).get("imageConfiguration", {})
        if current.get("imageUri") == desired_uri:
            return None
        if not await handler.stop_application_if_needed(application_id):
            return desired
        await handler.update_application(
            application_id=application_id,
            image_configuration=desired,
        )
        return None

    @staticmethod
    async def _authorize_existing_application(
        handler: EMRServerlessHandler,
        policy: EMRServerlessConnectorPolicy,
        application_id: str,
    ) -> None:
        application = await handler.get_application(application_id)
        if not policy.authorizes_application(application_id, application):
            raise ValueError(
                f"EMR Serverless application {application_id!r} is neither listed in "
                f"{_ENV_ALLOWED_APPLICATION_IDS} nor tagged as connector-owned under "
                f"a prefix configured by {_ENV_ALLOWED_APPLICATION_NAME_PREFIXES}"
            )

    @staticmethod
    def _execution_env(metadata: Optional[TaskExecutionMetadata]) -> Dict[str, str]:
        """Build Flyte 2 runtime variables absent from connector task pods."""
        if metadata is None:
            return {}
        env = {str(key): str(value) for key, value in metadata.environment_variables.items()}
        if metadata.HasField("task_execution_id"):
            task_execution_id = metadata.task_execution_id
            execution = task_execution_id.node_execution_id.execution_id
            task_id = task_execution_id.task_id
            derived = {
                "FLYTE_INTERNAL_EXECUTION_ID": execution.name,
                "FLYTE_INTERNAL_EXECUTION_PROJECT": execution.project,
                "FLYTE_INTERNAL_EXECUTION_DOMAIN": execution.domain,
                "_U_ORG_NAME": execution.org,
                "FLYTE_INTERNAL_TASK_PROJECT": task_id.project,
                "FLYTE_INTERNAL_TASK_DOMAIN": task_id.domain,
                "FLYTE_INTERNAL_TASK_NAME": task_id.name,
                "FLYTE_INTERNAL_TASK_VERSION": task_id.version,
                "FLYTE_INTERNAL_NODE_ID": task_execution_id.node_execution_id.node_id,
                "FLYTE_ATTEMPT_NUMBER": str(task_execution_id.retry_attempt),
                "FLYTE_INTERNAL_TASK_RETRY_ATTEMPT": str(task_execution_id.retry_attempt),
            }
            env = {**{key: value for key, value in derived.items() if value != ""}, **env}
        return env

    @classmethod
    def _format_env_as_spark_conf(cls, env_vars: Dict[str, str]) -> str:
        parts = []
        for key, value in env_vars.items():
            if _UNSAFE_ENV_VALUE_CHARS.search(value):
                logger.warning(
                    "Skipping Spark environment variable %s because its value is not safely representable", key
                )
                continue
            parts.extend(
                [
                    f"--conf spark.emr-serverless.driverEnv.{key}={value}",
                    f"--conf spark.executorEnv.{key}={value}",
                ]
            )
        return " ".join(parts)

    @classmethod
    def _append_flyte_env_to_spark_params(
        cls,
        existing: Optional[str],
        metadata: Optional[TaskExecutionMetadata],
        *,
        enabled: bool,
        template_env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        if not enabled:
            return existing
        env = {**(template_env or {}), **cls._execution_env(metadata)}
        rendered = cls._format_env_as_spark_conf(env)
        if not rendered:
            return existing
        return f"{existing or ''} {rendered}".strip()

    def _merge_spark_submit_params(self, user_params: Optional[str]) -> str:
        if not user_params:
            return self._PYTHONIC_SPARK_DEFAULTS
        additions = []
        for raw_item in self._PYTHONIC_SPARK_DEFAULTS.split("--conf "):
            item = raw_item.strip()
            if item and item.split("=", 1)[0] not in user_params:
                additions.append(f"--conf {item}")
        return f"{user_params} {' '.join(additions)}".strip()

    def _build_pythonic_job_driver(
        self,
        task_template: TaskTemplate,
        config: EMRServerless,
        handler: EMRServerlessHandler,
    ) -> Dict[str, Any]:
        container = task_template.container
        if container is None or not container.args:
            raise ValueError("Pythonic mode requires the task's Flyte 2 `a0` container command")
        args = list(container.args)
        if args[0] != "a0":
            raise ValueError(f"Pythonic mode expected the Flyte 2 `a0` entrypoint, got {args[0]!r}")
        return {
            "sparkSubmit": {
                "entryPoint": self._ensure_entrypoint_on_s3(handler, config),
                "entryPointArguments": args,
                "sparkSubmitParameters": self._merge_spark_submit_params(config.spark_submit_parameters),
            }
        }

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> EMRServerlessJobMetadata:
        """Resolve the application, request startup, and submit or defer the job."""
        config = self._extract_config(task_template)
        policy = self._get_policy()
        policy.validate_execution_role(config.execution_role_arn)
        handler = self._get_handler(config.region)
        region = config.region or handler.client.meta.region_name
        if not region:
            raise ValueError("Unable to resolve the AWS region for EMR Serverless")
        policy.validate_region(region)
        application_id = config.application_id
        created_application = False
        pending_image_configuration = None

        if application_id and not _APPLICATION_ID_RE.fullmatch(application_id):
            requested_name = application_id
            application_id = await handler.find_application_by_name(requested_name)
            if not application_id:
                raise ValueError(f"No active EMR Serverless application found with name {requested_name!r}")

        if not application_id:
            if not config.application_name:
                raise ValueError("No application_id provided and no application_name specified")
            if not self._is_create_application_allowed():
                raise ValueError(
                    f"Dynamic application creation is disabled by the connector; set "
                    f"{_ENV_ALLOW_CREATE_APPLICATION}=true or use an existing application_id"
                )
            application_id = await handler.create_application(
                name=self._apply_application_name_prefix(config.application_name),
                release_label=config.release_label,
                application_type=config.application_type,
                initial_capacity=config.initial_capacity,
                maximum_capacity=config.maximum_capacity,
                network_configuration=config.network_configuration,
                image_configuration=self._resolve_image_configuration(task_template, config, for_create=True),
                tags=self._merge_tags(config.tags),
                architecture=config.architecture,
                runtime_configuration=config.runtime_configuration,
                scheduler_configuration=config.scheduler_configuration,
                auto_stop_config=config.auto_stop_config,
            )
            created_application = True
        else:
            await self._authorize_existing_application(handler, policy, application_id)
            if config.sync_image:
                pending_image_configuration = await self._sync_application_image(
                    handler,
                    application_id,
                    task_template,
                    config,
                )

        if config.is_script_mode:
            job_driver = config.get_job_driver()
            if config.spark_submit_parameters and "sparkSubmit" in job_driver:
                existing = job_driver["sparkSubmit"].get("sparkSubmitParameters")
                job_driver["sparkSubmit"]["sparkSubmitParameters"] = (
                    f"{existing or ''} {config.spark_submit_parameters}".strip()
                )
        else:
            job_driver = self._build_pythonic_job_driver(task_template, config, handler)

        if "sparkSubmit" in job_driver:
            template_env = (
                {item.key: item.value for item in task_template.container.env} if task_template.container else {}
            )
            existing = job_driver["sparkSubmit"].get("sparkSubmitParameters")
            updated = self._append_flyte_env_to_spark_params(
                existing,
                task_execution_metadata,
                enabled=config.inject_flyte_env,
                template_env=template_env,
            )
            if updated:
                job_driver["sparkSubmit"]["sparkSubmitParameters"] = updated

        request: Dict[str, Any] = {
            "execution_role_arn": config.execution_role_arn,
            "job_driver": job_driver,
            "configuration_overrides": config.get_effective_configuration_overrides(),
            "tags": self._merge_tags(config.tags),
            "execution_timeout_minutes": config.execution_timeout_minutes,
            "name": task_template.id.name or None,
            "retry_policy": config.retry_policy,
        }
        request["client_token"] = config.client_token or _execution_token(task_execution_metadata, request)

        if pending_image_configuration or not await handler.start_application_if_needed(application_id):
            return EMRServerlessJobMetadata(
                application_id=application_id,
                region=region,
                created_application=created_application,
                is_script_mode=config.is_script_mode,
                pending_job_request=request,
                pending_image_configuration=pending_image_configuration,
            )
        return EMRServerlessJobMetadata(
            application_id=application_id,
            job_run_id=await handler.start_job_run(application_id=application_id, **request),
            region=region,
            created_application=created_application,
            is_script_mode=config.is_script_mode,
        )

    async def get(self, resource_meta: EMRServerlessJobMetadata, **kwargs: Any) -> Resource:
        """Return the current provider state."""
        handler = self._get_handler(resource_meta.region)
        job_run_id = resource_meta.job_run_id
        try:
            if not job_run_id:
                if not resource_meta.pending_job_request:
                    return Resource(phase=TaskExecution.FAILED, message="Job submission metadata is missing")
                if resource_meta.pending_image_configuration:
                    current = await handler.get_application(resource_meta.application_id)
                    desired_uri = resource_meta.pending_image_configuration.get("imageUri")
                    current_uri = current.get("imageConfiguration", {}).get("imageUri")
                    if current_uri != desired_uri:
                        if not await handler.stop_application_if_needed(resource_meta.application_id):
                            return Resource(
                                phase=TaskExecution.RUNNING,
                                message=(
                                    "EMR Serverless application "
                                    f"{resource_meta.application_id} is transitioning for image update"
                                ),
                            )
                        await handler.update_application(
                            application_id=resource_meta.application_id,
                            image_configuration=resource_meta.pending_image_configuration,
                        )
                if not await handler.start_application_if_needed(resource_meta.application_id):
                    return Resource(
                        phase=TaskExecution.RUNNING,
                        message=f"EMR Serverless application {resource_meta.application_id} is starting",
                    )
                job_run_id = await handler.start_job_run(
                    application_id=resource_meta.application_id,
                    **resource_meta.pending_job_request,
                )
            job = await handler.get_job_run(resource_meta.application_id, job_run_id)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            phase = TaskExecution.RETRYABLE_FAILED if code in _RETRYABLE_ERROR_CODES else TaskExecution.FAILED
            return Resource(phase=phase, message=str(error))
        except RuntimeError as error:
            return Resource(phase=TaskExecution.FAILED, message=str(error))

        state = job.get("state", "")
        if state in _JOB_RUNNING:
            phase = TaskExecution.RUNNING
        elif state == "SUCCESS":
            phase = TaskExecution.SUCCEEDED
        elif state == "CANCELLED":
            phase = TaskExecution.ABORTED
        elif state == "FAILED":
            phase = TaskExecution.FAILED
        else:
            return Resource(phase=TaskExecution.FAILED, message=f"Unrecognized EMR Serverless job state {state!r}")

        region = resource_meta.region or handler.client.meta.region_name or "us-east-1"
        details = job.get("stateDetails") or ""
        message = f"EMR Serverless job state: {state}" + (f" - {details}" if details else "")
        log = TaskLog(
            uri=(
                f"https://{region}.console.aws.amazon.com/emr/home?region={region}"
                f"#/serverless/{resource_meta.application_id}/jobs/{job_run_id}"
            ),
            name="EMR Serverless Console",
            ready=True,
            link_type=TaskLog.EXTERNAL,
        )
        return Resource(
            phase=phase,
            message=message,
            log_links=[log],
            outputs={} if phase == TaskExecution.SUCCEEDED and resource_meta.is_script_mode else None,
            custom_info={"application_id": resource_meta.application_id, "job_run_id": job_run_id},
        )

    async def delete(self, resource_meta: EMRServerlessJobMetadata, **kwargs: Any) -> None:
        """Cancel the exact idempotent job represented by the metadata."""
        handler = self._get_handler(resource_meta.region)
        job_run_id = resource_meta.job_run_id
        if not job_run_id and resource_meta.pending_image_configuration:
            return
        if not job_run_id and resource_meta.pending_job_request:
            if not await handler.start_application_if_needed(resource_meta.application_id):
                return
            job_run_id = await handler.start_job_run(
                application_id=resource_meta.application_id,
                **resource_meta.pending_job_request,
            )
        if job_run_id:
            await handler.cancel_job_run(resource_meta.application_id, job_run_id)
            logger.info("Cancelled EMR Serverless job %s", job_run_id)


# Compatibility with the original draft name.
EMRServerlessMetadata = EMRServerlessJobMetadata


ConnectorRegistry.register(EMRServerlessConnector())

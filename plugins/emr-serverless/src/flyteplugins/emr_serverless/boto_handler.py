"""Async boto3 operations used by the EMR Serverless connector."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "adaptive"})
_APPLICATION_READY = "STARTED"
_APPLICATION_STARTABLE = {"CREATED", "STOPPED"}
_APPLICATION_TRANSITIONING = {"CREATING", "STARTING", "STOPPING"}
_APPLICATION_TERMINAL = {"TERMINATED"}
_JOB_TERMINAL = {"SUCCESS", "FAILED", "CANCELLED"}


class EMRServerlessHandler:
    """Thin async wrapper around the boto3 EMR Serverless client.

    Args:
        region: AWS region, or `None` to use boto3's normal resolution.
    """

    def __init__(self, region: Optional[str] = None) -> None:
        self.region = region
        self._client: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Return the lazily created boto3 client."""
        if self._client is None:
            kwargs: Dict[str, Any] = {"service_name": "emr-serverless", "config": _BOTO_CONFIG}
            if self.region:
                kwargs["region_name"] = self.region
            self._client = boto3.client(**kwargs)
        return self._client

    async def _call(self, method: str, **params: Any) -> Dict[str, Any]:
        """Call one boto3 method without blocking the connector event loop."""
        return await asyncio.to_thread(getattr(self.client, method), **params)

    async def _paginate(self, method: str, result_key: str, **params: Any) -> List[Dict[str, Any]]:
        """Collect every item from a boto3 paginator."""

        def collect() -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for page in self.client.get_paginator(method).paginate(**params):
                items.extend(page.get(result_key, []))
            return items

        return await asyncio.to_thread(collect)

    async def get_application(self, application_id: str) -> Dict[str, Any]:
        """Return one application's AWS representation."""
        return (await self._call("get_application", applicationId=application_id)).get("application", {})

    async def find_application_by_name(self, name: str) -> Optional[str]:
        """Return the ID of an active application with an exact name match."""
        applications = await self._paginate(
            "list_applications",
            result_key="applications",
            states=[
                "CREATING",
                "CREATED",
                "STARTING",
                "STARTED",
                "STOPPING",
                "STOPPED",
            ],
        )
        for application in applications:
            if application.get("name") == name:
                return str(application["id"])
        return None

    async def create_application(
        self,
        *,
        name: str,
        release_label: str,
        application_type: str,
        initial_capacity: Optional[Dict[str, Any]] = None,
        maximum_capacity: Optional[Dict[str, Any]] = None,
        network_configuration: Optional[Dict[str, Any]] = None,
        image_configuration: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        architecture: Optional[str] = None,
        runtime_configuration: Optional[List[Dict[str, Any]]] = None,
        scheduler_configuration: Optional[Dict[str, Any]] = None,
        auto_stop_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create an application and return its identifier."""
        params: Dict[str, Any] = {
            "name": name,
            "releaseLabel": release_label,
            "type": application_type,
        }
        optional = {
            "initialCapacity": initial_capacity,
            "maximumCapacity": maximum_capacity,
            "networkConfiguration": network_configuration,
            "imageConfiguration": image_configuration,
            "tags": tags,
            "architecture": architecture,
            "runtimeConfiguration": runtime_configuration,
            "schedulerConfiguration": scheduler_configuration,
            "autoStopConfiguration": auto_stop_config,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        application_id = (await self._call("create_application", **params)).get("applicationId")
        if not application_id:
            raise RuntimeError("EMR Serverless CreateApplication response did not include applicationId")
        return str(application_id)

    async def update_application(
        self,
        *,
        application_id: str,
        image_configuration: Optional[Dict[str, Any]] = None,
        maximum_capacity: Optional[Dict[str, Any]] = None,
        auto_stop_config: Optional[Dict[str, Any]] = None,
        runtime_configuration: Optional[List[Dict[str, Any]]] = None,
        scheduler_configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update mutable application properties."""
        params: Dict[str, Any] = {"applicationId": application_id}
        optional = {
            "imageConfiguration": image_configuration,
            "maximumCapacity": maximum_capacity,
            "autoStopConfiguration": auto_stop_config,
            "runtimeConfiguration": runtime_configuration,
            "schedulerConfiguration": scheduler_configuration,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        if len(params) > 1:
            await self._call("update_application", **params)

    async def start_application_if_needed(self, application_id: str) -> bool:
        """Request startup and return whether the application is already ready."""
        state = (await self.get_application(application_id)).get("state", "")
        if state == _APPLICATION_READY:
            return True
        if state in _APPLICATION_TERMINAL:
            raise RuntimeError(f"Application {application_id} is in terminal state {state!r}")
        if state in _APPLICATION_STARTABLE:
            await self._call("start_application", applicationId=application_id)
            return False
        if state in _APPLICATION_TRANSITIONING:
            return False
        raise RuntimeError(f"Unrecognized EMR Serverless application state {state!r}")

    async def stop_application_if_needed(self, application_id: str) -> bool:
        """Request shutdown and return whether mutable settings can be updated."""
        state = (await self.get_application(application_id)).get("state", "")
        if state in _APPLICATION_STARTABLE:
            return True
        if state == _APPLICATION_READY:
            await self._call("stop_application", applicationId=application_id)
            return False
        if state in _APPLICATION_TRANSITIONING:
            return False
        if state in _APPLICATION_TERMINAL:
            raise RuntimeError(f"Application {application_id} is in terminal state {state!r}")
        raise RuntimeError(f"Unrecognized EMR Serverless application state {state!r}")

    async def ensure_application_started(
        self,
        application_id: str,
        timeout_seconds: int = 300,
        poll_interval_seconds: int = 10,
    ) -> None:
        """Wait until an application is ready.

        This blocking helper is retained for public Flytekit compatibility.
        Connector task creation uses `start_application_if_needed` so its RPC
        remains asynchronous.

        Args:
            application_id: Application to start.
            timeout_seconds: Maximum wait.
            poll_interval_seconds: Delay between provider reads.

        Raises:
            TimeoutError: The application did not start before the deadline.
            RuntimeError: The application entered an invalid or terminal state.
        """
        started_at = time.monotonic()
        while not await self.start_application_if_needed(application_id):
            if time.monotonic() - started_at >= timeout_seconds:
                raise TimeoutError(f"Timed out waiting for application {application_id} to start")
            await asyncio.sleep(poll_interval_seconds)

    async def start_job_run(
        self,
        *,
        application_id: str,
        execution_role_arn: str,
        job_driver: Dict[str, Any],
        configuration_overrides: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        execution_timeout_minutes: int = 60,
        name: Optional[str] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        client_token: Optional[str] = None,
    ) -> str:
        """Submit a job and return its run identifier."""
        params: Dict[str, Any] = {
            "applicationId": application_id,
            "executionRoleArn": execution_role_arn,
            "jobDriver": job_driver,
            "executionTimeoutMinutes": execution_timeout_minutes,
        }
        optional = {
            "configurationOverrides": configuration_overrides,
            "tags": tags,
            "name": name,
            "retryPolicy": retry_policy,
            "clientToken": client_token,
        }
        params.update({key: value for key, value in optional.items() if value is not None})
        job_run_id = (await self._call("start_job_run", **params)).get("jobRunId")
        if not job_run_id:
            raise RuntimeError("EMR Serverless StartJobRun response did not include jobRunId")
        return str(job_run_id)

    async def get_job_run(self, application_id: str, job_run_id: str) -> Dict[str, Any]:
        """Return one job run's AWS representation."""
        response = await self._call(
            "get_job_run",
            applicationId=application_id,
            jobRunId=job_run_id,
        )
        return response.get("jobRun", {})

    async def cancel_job_run(self, application_id: str, job_run_id: str) -> None:
        """Cancel a job idempotently."""
        try:
            state = (await self.get_job_run(application_id, job_run_id)).get("state", "")
            if state in _JOB_TERMINAL:
                return
            await self._call(
                "cancel_job_run",
                applicationId=application_id,
                jobRunId=job_run_id,
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code == "ResourceNotFoundException" or (
                code == "ValidationException" and "cannot be cancelled" in str(error).lower()
            ):
                return
            raise

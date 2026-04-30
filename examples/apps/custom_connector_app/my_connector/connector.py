"""
A simple custom connector that simulates submitting and polling an external batch job.

This demonstrates the AsyncConnector interface: create a job, poll its status,
and optionally cancel it. Replace the mock logic with real API calls for your
use case (e.g., a custom ML training service, a data pipeline, etc.).
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from flyteidl2.connector.connector_pb2 import GetTaskLogsResponse, GetTaskLogsResponseBody
from flyteidl2.core.execution_pb2 import TaskExecution

from flyte import logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta


@dataclass
class BatchJobMetadata(ResourceMeta):
    job_id: str
    created_at: float


class BatchJobConnector(AsyncConnector):
    """Simulates an external batch job service."""

    name = "Batch Job Connector"
    task_type_name = "batch_job"
    metadata_type = BatchJobMetadata

    async def create(
        self,
        task_template,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> BatchJobMetadata:
        job_id = str(uuid.uuid4())[:8]
        logger.info(f"Submitted batch job {job_id}")
        return BatchJobMetadata(job_id=job_id, created_at=time.time())

    async def get(self, resource_meta: BatchJobMetadata, **kwargs) -> Resource:
        elapsed = time.time() - resource_meta.created_at

        # Simulate a job that takes ~5 seconds to complete
        if elapsed < 5:
            logger.info(f"Job {resource_meta.job_id} still running ({elapsed:.0f}s)")
            return Resource(phase=TaskExecution.RUNNING, message="Job in progress")

        logger.info(f"Job {resource_meta.job_id} completed")
        return Resource(
            phase=TaskExecution.SUCCEEDED,
            message="Job completed successfully",
            outputs={"result": f"output-from-{resource_meta.job_id}"},
        )

    async def delete(self, resource_meta: BatchJobMetadata, **kwargs):
        logger.info(f"Cancelled job {resource_meta.job_id}")

    async def get_logs(self, resource_meta: BatchJobMetadata, **kwargs) -> GetTaskLogsResponse:
        logger.info(f"Fetching logs for job {resource_meta.job_id}")
        return GetTaskLogsResponse(
            body=GetTaskLogsResponseBody(
                results=[
                    f"[INFO] Job {resource_meta.job_id} started at {resource_meta.created_at}",
                    f"[INFO] Job {resource_meta.job_id} is processing...",
                    f"[INFO] Job {resource_meta.job_id} finished",
                ],
            ),
        )


ConnectorRegistry.register(BatchJobConnector())

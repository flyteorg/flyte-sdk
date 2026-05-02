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

from flyteidl2.connector.connector_pb2 import (
    GetTaskLogsResponse,
    GetTaskLogsResponseBody,
    GetTaskLogsResponseHeader,
)
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.logs.dataplane.payload_pb2 import LogLine, LogLineOriginator
from google.protobuf.timestamp_pb2 import Timestamp

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

    async def get_logs(self, resource_meta: BatchJobMetadata, token: str = "", **kwargs):
        logger.info(f"Fetching logs for job {resource_meta.job_id} (token={token!r})")

        def line(message: str, ts: float) -> LogLine:
            t = Timestamp()
            t.FromSeconds(int(ts))
            return LogLine(timestamp=t, message=message, originator=LogLineOriginator.USER)

        start = resource_meta.created_at
        job_id = resource_meta.job_id
        pages = {
            "": GetTaskLogsResponseBody(
                lines=[
                    line(f"[INFO] Job {job_id} submitted", start),
                    line(f"[INFO] Job {job_id} started", start + 1),
                ],
            ),
            "page-2": GetTaskLogsResponseBody(
                lines=[
                    line(f"[INFO] Job {job_id} is processing...", start + 2),
                    line(f"[INFO] Job {job_id} step 1 complete", start + 3),
                ],
            ),
            "page-3": GetTaskLogsResponseBody(
                lines=[
                    line(f"[INFO] Job {job_id} step 2 complete", start + 4),
                    line(f"[INFO] Job {job_id} finished", start + 5),
                ],
            ),
        }
        next_tokens = {"": "page-2", "page-2": "page-3", "page-3": ""}

        body = pages.get(token, GetTaskLogsResponseBody(lines=[]))
        next_token = next_tokens.get(token, "")

        # Body first, then a header carrying the next-page token (omitted on
        # the final page so the operator stops paging).
        yield GetTaskLogsResponse(body=body)
        if next_token:
            yield GetTaskLogsResponse(
                header=GetTaskLogsResponseHeader(token=next_token),
            )


ConnectorRegistry.register(BatchJobConnector())

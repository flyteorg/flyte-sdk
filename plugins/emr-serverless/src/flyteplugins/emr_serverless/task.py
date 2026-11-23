from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from flyte._task_plugins import TaskPluginRegistry
from flyte.connectors import AsyncConnectorExecutorMixin
from flyte.extend import AsyncFunctionTaskTemplate
from flyte.models import SerializationContext

TASK_TYPE = "emr_serverless"


@dataclass
class EMRServerless:
    """Configuration for submitting an existing Spark or Hive job to EMR Serverless.

    The request fields map directly to the AWS `StartJobRun` API. The connector
    starts a stopped application without blocking task creation, then submits the
    job once the application is ready.

    Args:
        application_id: Existing EMR Serverless application identifier.
        execution_role_arn: IAM role used by EMR Serverless for the job.
        job_driver: Spark or Hive `StartJobRun.jobDriver` payload.
        region: AWS region. When omitted, the boto3 default resolution applies.
        configuration_overrides: Optional `StartJobRun.configurationOverrides`.
        tags: Optional tags added to the job run.
        execution_timeout_minutes: Maximum job duration in minutes.
        name: Optional job name. Flyte's task name is used when omitted.
        retry_policy: Optional EMR Serverless job retry policy.
        client_token: Optional idempotency token. Flyte derives one from the
            task execution identifier when omitted.
    """

    application_id: str
    execution_role_arn: str
    job_driver: Dict[str, Any]
    region: Optional[str] = None
    configuration_overrides: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    execution_timeout_minutes: int = 60
    name: Optional[str] = None
    retry_policy: Optional[Dict[str, Any]] = None
    client_token: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.application_id:
            raise ValueError("application_id is required")
        if not self.execution_role_arn:
            raise ValueError("execution_role_arn is required")
        if set(self.job_driver) not in ({"sparkSubmit"}, {"hive"}):
            raise ValueError("job_driver must contain exactly one of 'sparkSubmit' or 'hive'")
        if self.execution_timeout_minutes < 1:
            raise ValueError("execution_timeout_minutes must be at least 1")

    def to_custom_config(self, task_name: str) -> Dict[str, Any]:
        """Serialize the configuration into the connector's provider-neutral contract."""
        request: Dict[str, Any] = {
            "applicationId": self.application_id,
            "executionRoleArn": self.execution_role_arn,
            "jobDriver": self.job_driver,
            "executionTimeoutMinutes": self.execution_timeout_minutes,
            "name": self.name or task_name,
        }
        if self.configuration_overrides:
            request["configurationOverrides"] = self.configuration_overrides
        if self.tags:
            request["tags"] = dict(self.tags)
        if self.retry_policy:
            request["retryPolicy"] = self.retry_policy
        if self.client_token:
            request["clientToken"] = self.client_token
        return {
            "region": self.region,
            "request": request,
        }


class EMRServerlessFunctionTask(AsyncConnectorExecutorMixin, AsyncFunctionTaskTemplate):
    """A Flyte task represented by an EMR Serverless `StartJobRun` request.

    The decorated function defines the Flyte task interface but is not executed
    by the connector. Script and Hive jobs should therefore declare no Flyte
    outputs; provider status and console links remain available on the action.
    """

    plugin_config: EMRServerless

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task_type = TASK_TYPE

    def custom_config(self, sctx: SerializationContext) -> Dict[str, Any]:
        """Return the serialized EMR Serverless request."""
        if self.interface.outputs:
            raise ValueError(
                "EMR Serverless script tasks cannot declare Flyte outputs; "
                "write results to durable storage and return no values"
            )
        return self.plugin_config.to_custom_config(self.name)


TaskPluginRegistry.register(EMRServerless, EMRServerlessFunctionTask)

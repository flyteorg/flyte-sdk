from __future__ import annotations

import os
from urllib.parse import urlparse

# Set environment variables for gRPC, this reduces log spew and avoids unnecessary warnings
# before importing grpc
if "GRPC_VERBOSITY" not in os.environ:
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GRPC_CPP_MIN_LOG_LEVEL"] = "ERROR"
    # Disable fork support (stops "skipping fork() handlers")
    os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
    # Reduce absl/glog verbosity
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["ABSL_LOG"] = "0"
#### Has to be before grpc

import grpc
from flyteidl.service import admin_pb2_grpc, dataproxy_pb2_grpc, identity_pb2_grpc
from flyteidl2.app import app_service_pb2_grpc
from flyteidl2.secret import secret_pb2_grpc
from flyteidl2.task import task_service_pb2_grpc
from flyteidl2.trigger import trigger_service_pb2_grpc
from flyteidl2.workflow import run_logs_service_pb2_grpc, run_service_pb2_grpc

from ._protocols import (
    AppService,
    DataProxyService,
    IdentityService,
    MetadataServiceProtocol,
    ProjectDomainService,
    RunLogsService,
    RunService,
    SecretService,
    TaskService,
    TriggerService,
)
from .auth import create_channel


class Console:
    """
    Console URL builder for Flyte resources.

    Constructs console URLs for various Flyte resources (tasks, runs, apps, triggers)
    based on the configured endpoint and security settings.

    Args:
        endpoint: The Flyte endpoint (e.g., "dns:///localhost:8090", "https://example.com")
        insecure: Whether to use HTTP (True) or HTTPS (False)

    Example:
        >>> console = Console("dns:///example.com", insecure=False)
        >>> url = console.task_url(project="myproject", domain="development", task_name="mytask")
    """

    def __init__(self, endpoint: str, insecure: bool = False):
        """
        Initialize Console with endpoint and security configuration.

        Args:
            endpoint: The Flyte endpoint URL
            insecure: Whether to use HTTP (True) or HTTPS (False)
        """
        self._endpoint = endpoint
        self._insecure = insecure
        self._http_domain = self._compute_http_domain()

    def _compute_http_domain(self) -> str:
        """
        Compute the HTTP domain from the endpoint.

        Internal method that extracts and normalizes the domain from various
        endpoint formats (dns://, http://, https://).

        Returns:
            The normalized HTTP(S) domain URL
        """
        scheme = "http" if self._insecure else "https"
        parsed = urlparse(self._endpoint)
        if parsed.scheme == "dns":
            domain = parsed.path.lstrip("/")
        else:
            domain = parsed.netloc or parsed.path

        # TODO: make console url configurable
        domain_split = domain.split(":")
        if domain_split[0] == "localhost":
            # Always use port 8080 for localhost, until the to do is done.
            domain = "localhost:8080"

        return f"{scheme}://{domain}"

    def _resource_url(self, project: str, domain: str, resource: str, resource_name: str) -> str:
        """
        Internal helper to build a resource URL.

        Args:
            project: Project name
            domain: Domain name
            resource: Resource type (e.g., "tasks", "runs", "apps", "triggers")
            resource_name: Resource identifier

        Returns:
            The full console URL for the resource
        """
        return f"{self._http_domain}/v2/domain/{domain}/project/{project}/{resource}/{resource_name}"

    def run_url(self, project: str, domain: str, run_name: str) -> str:
        """
        Build console URL for a run.

        Args:
            project: Project name
            domain: Domain name
            run_name: Run identifier

        Returns:
            Console URL for the run
        """
        return self._resource_url(project, domain, "runs", run_name)

    def app_url(self, project: str, domain: str, app_name: str) -> str:
        """
        Build console URL for an app.

        Args:
            project: Project name
            domain: Domain name
            app_name: App identifier

        Returns:
            Console URL for the app
        """
        return self._resource_url(project, domain, "apps", app_name)

    def task_url(self, project: str, domain: str, task_name: str) -> str:
        """
        Build console URL for a task.

        Args:
            project: Project name
            domain: Domain name
            task_name: Task identifier

        Returns:
            Console URL for the task
        """
        return self._resource_url(project, domain, "tasks", task_name)

    def trigger_url(self, project: str, domain: str, task_name: str, trigger_name: str) -> str:
        """
        Build console URL for a trigger.

        Args:
            project: Project name
            domain: Domain name
            task_name: Task identifier
            trigger_name: Trigger identifier

        Returns:
            Console URL for the trigger
        """
        return self._resource_url(project, domain, "triggers", f"{task_name}/{trigger_name}")

    @property
    def endpoint(self) -> str:
        """The configured endpoint."""
        return self._endpoint

    @property
    def insecure(self) -> bool:
        """Whether insecure (HTTP) mode is enabled."""
        return self._insecure


class ClientSet:
    def __init__(
        self,
        channel: grpc.aio.Channel,
        endpoint: str,
        insecure: bool = False,
        **kwargs,
    ):
        self.endpoint = endpoint
        self.insecure = insecure
        self._channel = channel
        self._console = Console(self.endpoint, self.insecure)
        self._admin_client = admin_pb2_grpc.AdminServiceStub(channel=channel)
        self._task_service = task_service_pb2_grpc.TaskServiceStub(channel=channel)
        self._app_service = app_service_pb2_grpc.AppServiceStub(channel=channel)
        self._run_service = run_service_pb2_grpc.RunServiceStub(channel=channel)
        self._dataproxy = dataproxy_pb2_grpc.DataProxyServiceStub(channel=channel)
        self._log_service = run_logs_service_pb2_grpc.RunLogsServiceStub(channel=channel)
        self._secrets_service = secret_pb2_grpc.SecretServiceStub(channel=channel)
        self._identity_service = identity_pb2_grpc.IdentityServiceStub(channel=channel)
        self._trigger_service = trigger_service_pb2_grpc.TriggerServiceStub(channel=channel)

    @classmethod
    async def for_endpoint(cls, endpoint: str, *, insecure: bool = False, **kwargs) -> ClientSet:
        return cls(
            await create_channel(endpoint, None, insecure=insecure, **kwargs), endpoint, insecure=insecure, **kwargs
        )

    @classmethod
    async def for_api_key(cls, api_key: str, *, insecure: bool = False, **kwargs) -> ClientSet:
        from flyte.remote._client.auth._auth_utils import decode_api_key

        # Parsing the API key is done in create_channel, but cleaner to redo it here rather than getting create_channel
        # to return the endpoint
        endpoint, _, _, _ = decode_api_key(api_key)

        return cls(
            await create_channel(None, api_key, insecure=insecure, **kwargs), endpoint, insecure=insecure, **kwargs
        )

    @classmethod
    async def for_serverless(cls) -> ClientSet:
        raise NotImplementedError

    @classmethod
    async def from_env(cls) -> ClientSet:
        raise NotImplementedError

    @property
    def metadata_service(self) -> MetadataServiceProtocol:
        return self._admin_client

    @property
    def project_domain_service(self) -> ProjectDomainService:
        return self._admin_client

    @property
    def task_service(self) -> TaskService:
        return self._task_service

    @property
    def app_service(self) -> AppService:
        return self._app_service

    @property
    def run_service(self) -> RunService:
        return self._run_service

    @property
    def dataproxy_service(self) -> DataProxyService:
        return self._dataproxy

    @property
    def logs_service(self) -> RunLogsService:
        return self._log_service

    @property
    def secrets_service(self) -> SecretService:
        return self._secrets_service

    @property
    def identity_service(self) -> IdentityService:
        return self._identity_service

    @property
    def trigger_service(self) -> TriggerService:
        return self._trigger_service

    @property
    def console(self) -> Console:
        """
        Get the Console instance for this client.

        Returns a Console configured with this client's endpoint and security settings.
        Use this to build console URLs for Flyte resources.

        Returns:
            Console instance

        Example:
            >>> client = get_client()
            >>> url = client.console.task_url(project="myproj", domain="dev", task_name="mytask")
        """
        return self._console

    async def close(self, grace: float | None = None):
        return await self._channel.close(grace=grace)

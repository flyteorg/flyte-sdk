from __future__ import annotations

from urllib.parse import urlparse

from flyteidl2.app.app_service_connect import AppServiceClient
from flyteidl2.auth.identity_connect import IdentityServiceClient
from flyteidl2.dataproxy.dataproxy_service_connect import DataProxyServiceClient
from flyteidl2.project.project_service_connect import ProjectServiceClient
from flyteidl2.secret.secret_connect import SecretServiceClient
from flyteidl2.task.task_service_connect import TaskServiceClient
from flyteidl2.trigger.trigger_service_connect import TriggerServiceClient
from flyteidl2.workflow.run_logs_service_connect import RunLogsServiceClient
from flyteidl2.workflow.run_service_connect import RunServiceClient

from ._protocols import (
    AppService,
    DataProxyService,
    IdentityService,
    ProjectDomainService,
    RunLogsService,
    RunService,
    SecretService,
    TaskService,
    TriggerService,
)
from .auth._session import SessionConfig, create_session_config


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
        host, _, port = domain.partition(":")
        if host == "localhost" and port == "8090":
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
    def __init__(self, session_cfg: SessionConfig):
        self._console = Console(session_cfg.endpoint, session_cfg.insecure)
        self._session_config = session_cfg
        shared = session_cfg.connect_kwargs()
        self._admin_client = ProjectServiceClient(**shared)
        self._task_service = TaskServiceClient(**shared)
        self._app_service = AppServiceClient(**shared)
        self._run_service = RunServiceClient(**shared)
        self._dataproxy = DataProxyServiceClient(**shared)
        self._log_service = RunLogsServiceClient(**shared)
        self._secrets_service = SecretServiceClient(**shared)
        self._identity_service = IdentityServiceClient(**shared)
        self._trigger_service = TriggerServiceClient(**shared)

    @classmethod
    async def for_endpoint(cls, endpoint: str, *, insecure: bool = False, **kwargs) -> ClientSet:
        rpc_retries = kwargs.pop("rpc_retries", None)
        session_cfg = await create_session_config(endpoint, None, insecure=insecure, rpc_retries=rpc_retries, **kwargs)
        return cls(session_cfg)

    @classmethod
    async def for_api_key(cls, api_key: str, *, insecure: bool = False, **kwargs) -> ClientSet:
        rpc_retries = kwargs.pop("rpc_retries", None)
        session_cfg = await create_session_config(None, api_key, insecure=insecure, rpc_retries=rpc_retries, **kwargs)
        return cls(session_cfg)

    @classmethod
    async def for_serverless(cls) -> ClientSet:
        raise NotImplementedError

    @classmethod
    async def from_env(cls) -> ClientSet:
        raise NotImplementedError

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
    def endpoint(self) -> str:
        return self._session_config.endpoint

    @property
    def session_config(self) -> SessionConfig:
        """The session configuration used by this client.

        Useful for external packages that need to create their own ConnectRPC
        service clients sharing the same transport and auth interceptors.
        """
        return self._session_config

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

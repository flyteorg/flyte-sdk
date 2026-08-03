from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar, cast
from urllib.parse import urlparse

from async_lru import alru_cache
from connectrpc.errors import ConnectError
from flyteidl2.app.app_service_connect import AppServiceClient
from flyteidl2.auth.identity_connect import IdentityServiceClient
from flyteidl2.cluster import payload_pb2 as cluster_payload_pb2
from flyteidl2.cluster.service_connect import ClusterServiceClient
from flyteidl2.common import identifier_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.dataproxy.dataproxy_service_connect import DataProxyServiceClient
from flyteidl2.imagebuilder import payload_pb2 as image_payload_pb2
from flyteidl2.imagebuilder.service_connect import ImageServiceClient
from flyteidl2.project.project_service_connect import ProjectServiceClient
from flyteidl2.secret import payload_pb2 as secret_payload_pb2
from flyteidl2.secret.secret_connect import SecretServiceClient
from flyteidl2.settings.settings_service_connect import SettingsServiceClient
from flyteidl2.task.task_service_connect import TaskServiceClient
from flyteidl2.trigger.trigger_service_connect import TriggerServiceClient
from flyteidl2.workflow.local_run_service_connect import LocalRunServiceClient
from flyteidl2.workflow.run_logs_service_connect import RunLogsServiceClient
from flyteidl2.workflow.run_service_connect import RunServiceClient

from ._protocols import (
    AppService,
    ClusterService,
    DataProxyService,
    IdentityService,
    ImageService,
    LocalRunService,
    ProjectDomainService,
    RunLogsService,
    RunService,
    SecretService,
    SettingsService,
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

    def local_run_url(self, project: str, domain: str, run_name: str) -> str:
        """
        Build console URL for a local run (a run orchestrated on the user's machine
        whose state is reported to the control plane via LocalRunService).

        Args:
            project: Project name
            domain: Domain name
            run_name: Run identifier

        Returns:
            Console URL for the local run
        """
        return self._resource_url(project, domain, "local-runs", run_name)

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


class _ClusterAwareService:
    """Shared machinery for the cluster-aware service wrappers.

    Each control-plane service below (dataproxy, secrets, images) must route every
    call to the cluster that ``ClusterService.SelectCluster`` picks for the target
    resource. The per-subclass part is just *which* connectrpc client class to build
    and *what* to call it in logs; the SelectCluster call, the same-endpoint
    short-circuit, and the auth-kwarg-preserving per-cluster session build are
    identical, so they live here.

    Subclasses provide:
      * ``_new_client`` — construct the connectrpc ``*ServiceClient`` for a
        resolved cluster endpoint.
      * ``_label`` — a human name used in debug logs.
      * ``_reraise_connect_error`` — when True, a ``ConnectError`` from SelectCluster
        propagates unwrapped so callers can branch on its gRPC code (the dataproxy
        ``OPERATION_UPLOAD_TRIGGER`` fallback relies on this); otherwise every
        failure is wrapped in ``RuntimeError``.
    """

    _label: ClassVar[str]
    _reraise_connect_error: ClassVar[bool] = False

    def __init__(
        self,
        cluster_service: ClusterService,
        session_config: SessionConfig,
        default_client: Any,
    ):
        self._cluster_service = cluster_service
        self._session_config = session_config
        self._default_client = default_client

    def _new_client(self, **connect_kwargs: Any) -> Any:
        """Construct a per-cluster connectrpc client. Overridden per service."""
        raise NotImplementedError

    async def _select_and_build(self, req: cluster_payload_pb2.SelectClusterRequest) -> Any:
        """SelectCluster + build the per-cluster client.

        Wrapped by the ``@alru_cache`` resolvers on each subclass; ``@alru_cache``
        deduplicates concurrent callers and only caches successful results, so a
        transient failure won't poison the entry.
        """
        client, _ = await self._select_and_build_with_cluster(req)
        return client

    async def _select_and_build_with_cluster(self, req: cluster_payload_pb2.SelectClusterRequest) -> tuple[Any, str]:
        """SelectCluster + build the per-cluster client, returning ``(client, cluster)``.

        ``cluster`` is the SelectCluster response's cluster name, or ``""`` when the
        call is served by the control plane (no endpoint returned, or the endpoint is
        the session's own — the same-endpoint short-circuit).
        """
        from flyte._logging import logger

        op_name = cluster_payload_pb2.SelectClusterRequest.Operation.Name(req.operation)
        try:
            resp = await self._cluster_service.select_cluster(req)
        except Exception as e:
            # Preserve the gRPC code (e.g. UNIMPLEMENTED for an unsupported operation)
            # for callers that branch on it — notably the dataproxy
            # OPERATION_UPLOAD_TRIGGER fallback to inline inputs.
            if self._reraise_connect_error and isinstance(e, ConnectError):
                raise
            raise RuntimeError(f"SelectCluster failed for {op_name}: {e}") from e

        endpoint = resp.cluster_endpoint
        if not endpoint or endpoint == self._session_config.endpoint:
            return self._default_client, ""

        # Forward the auth-related kwargs from the parent SessionConfig so the
        # per-cluster session preserves the configured ``auth_type`` (Passthrough,
        # ClientSecret, ExternalCommand, etc.). Without this ``create_session_config``
        # falls back to the default ``auth_type="Pkce"`` and a Passthrough-only
        # caller (e.g. a FastAPI app using ``init_passthrough``) trips the PKCE
        # browser flow as soon as the first cluster-routed call runs.
        auth_kwargs = dict(self._session_config.auth_kwargs or {})
        try:
            new_cfg = await create_session_config(
                endpoint,
                self._session_config.api_key,
                insecure=self._session_config.insecure,
                insecure_skip_verify=self._session_config.insecure_skip_verify,
                auth_endpoint=self._session_config.endpoint,
                **auth_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create session for cluster endpoint '{endpoint}': {e}") from e

        logger.debug(f"Created {self._label} client for cluster endpoint: {endpoint}")
        return self._new_client(**new_cfg.connect_kwargs()), resp.cluster


class ClusterAwareDataProxy(_ClusterAwareService):
    """DataProxy client that routes each call to the correct cluster.

    Implements the DataProxyService protocol. For every RPC, extracts the target
    resource from the request, calls ClusterService.SelectCluster to discover
    the cluster endpoint, and dispatches to a DataProxyServiceClient pointing at
    that endpoint. Per-cluster clients are cached by (operation, resource) so
    repeated calls against the same resource reuse the same connection.
    """

    _label = "DataProxy"
    # Preserve ConnectError so callers can branch on the gRPC code (the
    # OPERATION_UPLOAD_TRIGGER fallback to inline inputs depends on this).
    _reraise_connect_error = True

    def _new_client(self, **connect_kwargs: Any) -> DataProxyService:
        return cast(DataProxyService, DataProxyServiceClient(**connect_kwargs))

    async def create_upload_location(
        self, request: dataproxy_service_pb2.CreateUploadLocationRequest
    ) -> dataproxy_service_pb2.CreateUploadLocationResponse:
        client = await self._resolve(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_UPLOAD_LOCATION),
            request.org,
            request.project,
            request.domain,
        )
        return await client.create_upload_location(request)

    async def upload_inputs(
        self, request: dataproxy_service_pb2.UploadInputsRequest
    ) -> dataproxy_service_pb2.UploadInputsResponse:
        which = request.WhichOneof("id")
        if which == "run_id":
            # SelectClusterRequest.resource doesn't include RunIdentifier; route by project.
            org, project, domain = request.run_id.org, request.run_id.project, request.run_id.domain
        elif which == "project_id":
            org = request.project_id.organization
            project = request.project_id.name
            domain = request.project_id.domain
        else:
            raise ValueError("UploadInputsRequest must set either run_id or project_id")
        client = await self._resolve(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_UPLOAD_INPUTS),
            org,
            project,
            domain,
        )
        return await client.upload_inputs(request)

    async def upload_trigger(
        self, request: dataproxy_service_pb2.UploadInputsRequest
    ) -> dataproxy_service_pb2.UploadInputsResponse:
        """Upload trigger inputs, routing via SelectCluster's OPERATION_UPLOAD_TRIGGER.

        The actual upload is the same UploadInputs RPC; only the cluster-selection operation differs,
        so zero-trust backends can route trigger uploads to the data plane. When zero-trust is not
        enabled the backend returns UNIMPLEMENTED for this operation, which propagates to the caller
        (`trigger_serde.offload_trigger_inputs`) so it can fall back to inline trigger inputs.
        """
        which = request.WhichOneof("id")
        if which == "run_id":
            org, project, domain = request.run_id.org, request.run_id.project, request.run_id.domain
        elif which == "project_id":
            org = request.project_id.organization
            project = request.project_id.name
            domain = request.project_id.domain
        else:
            raise ValueError("UploadInputsRequest must set either run_id or project_id")
        client = await self._resolve(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_UPLOAD_TRIGGER),
            org,
            project,
            domain,
        )
        return await client.upload_inputs(request)

    async def get_action_data(
        self, request: dataproxy_service_pb2.GetActionDataRequest
    ) -> dataproxy_service_pb2.GetActionDataResponse:
        run = request.action_id.run
        client = await self._resolve_by_action(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_ACTION_DATA),
            run.org,
            run.project,
            run.domain,
            run.name,
            request.action_id.name,
        )
        return await client.get_action_data(request)

    async def create_download_link(
        self, request: dataproxy_service_pb2.CreateDownloadLinkRequest
    ) -> dataproxy_service_pb2.CreateDownloadLinkResponse:
        which = request.WhichOneof("source")
        if which == "action_attempt_id":
            run = request.action_attempt_id.action_id.run
            client = await self._resolve_by_action(
                int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_DOWNLOAD_LINK),
                run.org,
                run.project,
                run.domain,
                run.name,
                request.action_attempt_id.action_id.name,
            )
        elif which == "task_id":
            client = await self._resolve(
                int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_CREATE_DOWNLOAD_LINK),
                request.task_id.org,
                request.task_id.project,
                request.task_id.domain,
            )
        else:
            # app_id (or unset): route via the default client.
            client = self._default_client
        return await client.create_download_link(request)

    async def create_local_run_upload_location(
        self, request: dataproxy_service_pb2.CreateUploadLocationRequest
    ) -> tuple[dataproxy_service_pb2.CreateUploadLocationResponse, str]:
        """Signed upload URL for a local run's metadata artifact (inputs.pb / outputs.pb / report.html).

        Routes via SelectCluster's ``OPERATION_LOCAL_RUN_DATA`` so backends can direct
        local-run artifacts at a dataplane's storage. Returns ``(response, cluster)``
        where ``cluster`` is the routing cluster's name — ``""`` when the upload is
        served by the control plane — so callers can stamp it on reported attempt
        events and later reads route to the same cluster.
        """
        client, cluster = await self._resolve_with_cluster(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_LOCAL_RUN_DATA),
            request.org,
            request.project,
            request.domain,
        )
        return await client.create_upload_location(request), cluster

    def tail_logs(
        self, request: dataproxy_service_pb2.TailLogsRequest
    ) -> AsyncIterator[dataproxy_service_pb2.TailLogsResponse]:
        return self._tail_logs(request)

    async def _tail_logs(
        self, request: dataproxy_service_pb2.TailLogsRequest
    ) -> AsyncIterator[dataproxy_service_pb2.TailLogsResponse]:
        run = request.action_id.run
        client = await self._resolve_by_action(
            int(cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_TAIL_LOGS),
            run.org,
            run.project,
            run.domain,
            run.name,
            request.action_id.name,
        )
        async for resp in client.tail_logs(request):
            yield resp

    @alru_cache
    async def _resolve(self, operation: int, org: str, project: str, domain: str) -> DataProxyService:
        """Cached SelectCluster lookup, routed by ProjectIdentifier."""
        req = cluster_payload_pb2.SelectClusterRequest(operation=operation)
        req.project_id.CopyFrom(identifier_pb2.ProjectIdentifier(name=project, domain=domain, organization=org))
        return await self._select_and_build(req)

    @alru_cache
    async def _resolve_with_cluster(
        self, operation: int, org: str, project: str, domain: str
    ) -> tuple[DataProxyService, str]:
        """Cached SelectCluster lookup, routed by ProjectIdentifier, that also returns
        the selected cluster's name ("" when served by the control plane)."""
        req = cluster_payload_pb2.SelectClusterRequest(operation=operation)
        req.project_id.CopyFrom(identifier_pb2.ProjectIdentifier(name=project, domain=domain, organization=org))
        return await self._select_and_build_with_cluster(req)

    @alru_cache
    async def _resolve_by_action(
        self,
        operation: int,
        org: str,
        project: str,
        domain: str,
        run_name: str,
        action_name: str,
    ) -> DataProxyService:
        """Cached SelectCluster lookup, routed by ActionIdentifier."""
        req = cluster_payload_pb2.SelectClusterRequest(operation=operation)
        req.action_id.CopyFrom(
            identifier_pb2.ActionIdentifier(
                run=identifier_pb2.RunIdentifier(org=org, project=project, domain=domain, name=run_name),
                name=action_name,
            )
        )
        return await self._select_and_build(req)


class ClusterAwareSecretService(_ClusterAwareService):
    """Secret service client that routes each call to the correct cluster.

    Same pattern as ClusterAwareDataProxy: uses SelectCluster with
    OPERATION_USE_SECRETS to discover the cluster endpoint, then dispatches
    to a per-cluster SecretServiceClient. Clients are cached by project.
    """

    _label = "SecretService"

    def _new_client(self, **connect_kwargs: Any) -> SecretService:
        return cast(SecretService, SecretServiceClient(**connect_kwargs))

    async def create_secret(
        self, request: secret_payload_pb2.CreateSecretRequest
    ) -> secret_payload_pb2.CreateSecretResponse:
        client = await self._resolve(request.id.organization, request.id.project, request.id.domain)
        return await client.create_secret(request)

    async def update_secret(
        self, request: secret_payload_pb2.UpdateSecretRequest
    ) -> secret_payload_pb2.UpdateSecretResponse:
        client = await self._resolve(request.id.organization, request.id.project, request.id.domain)
        return await client.update_secret(request)

    async def get_secret(self, request: secret_payload_pb2.GetSecretRequest) -> secret_payload_pb2.GetSecretResponse:
        client = await self._resolve(request.id.organization, request.id.project, request.id.domain)
        return await client.get_secret(request)

    async def list_secrets(
        self, request: secret_payload_pb2.ListSecretsRequest
    ) -> secret_payload_pb2.ListSecretsResponse:
        client = await self._resolve(request.organization, request.project, request.domain)
        return await client.list_secrets(request)

    async def delete_secret(
        self, request: secret_payload_pb2.DeleteSecretRequest
    ) -> secret_payload_pb2.DeleteSecretResponse:
        client = await self._resolve(request.id.organization, request.id.project, request.id.domain)
        return await client.delete_secret(request)

    async def client_for_cluster_pool(self, org: str, name: str) -> SecretService:
        """Resolve a per-cluster SecretService client for a cluster pool.

        Used by SDK callers (e.g. flyte.remote.Secret) when an operation is scoped
        to a cluster pool rather than to a project/domain/org. cluster_pool is
        SDK-side routing metadata only — it is passed to SelectCluster but is not
        carried in the secret request proto, since the resolved cluster's secret
        service does not need it.
        """
        return await self._resolve(org, "", "", name)

    @alru_cache
    async def _resolve(self, org: str, project: str, domain: str, cluster_pool: str | None = None) -> SecretService:
        """Cached SelectCluster lookup for secrets.

        Routes by ClusterPoolIdentifier when cluster_pool is set;
        otherwise by ProjectIdentifier when project and domain are set,
        DomainIdentifier when only domain is set (domain-scoped secrets),
        or OrgIdentifier for org-wide secrets.
        """
        req = cluster_payload_pb2.SelectClusterRequest(
            operation=cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_USE_SECRETS,
        )
        if cluster_pool:
            req.cluster_pool_id.CopyFrom(identifier_pb2.ClusterPoolIdentifier(organization=org, name=cluster_pool))
        elif project and domain:
            req.project_id.CopyFrom(identifier_pb2.ProjectIdentifier(name=project, domain=domain, organization=org))
        elif domain:
            req.domain_id.CopyFrom(identifier_pb2.DomainIdentifier(name=domain, organization=org))
        else:
            req.org_id.CopyFrom(identifier_pb2.OrgIdentifier(name=org))
        return await self._select_and_build(req)


class ClusterAwareImageService(_ClusterAwareService):
    """Image service client that routes each call to the correct cluster.

    Same pattern as ClusterAwareDataProxy: uses SelectCluster with
    OPERATION_GET_IMAGE to discover the cluster endpoint, then dispatches to a
    per-cluster ImageServiceClient. Clients are cached by project.
    """

    _label = "ImageService"

    def _new_client(self, **connect_kwargs: Any) -> ImageService:
        return cast(ImageService, ImageServiceClient(**connect_kwargs))

    async def get_image(self, request: image_payload_pb2.GetImageRequest) -> image_payload_pb2.GetImageResponse:
        org = request.project_id.organization or request.organization
        client = await self._resolve(org, request.project_id.name, request.project_id.domain)
        return await client.get_image(request)

    @alru_cache
    async def _resolve(self, org: str, project: str, domain: str) -> ImageService:
        """Cached SelectCluster lookup for image reads.

        Routes by ProjectIdentifier when project and domain are set, falling
        back to OrgIdentifier (the backend then applies its default
        project/domain for image-builder resources).
        """
        req = cluster_payload_pb2.SelectClusterRequest(
            operation=cluster_payload_pb2.SelectClusterRequest.Operation.OPERATION_GET_IMAGE,
        )
        if project and domain:
            req.project_id.CopyFrom(identifier_pb2.ProjectIdentifier(name=project, domain=domain, organization=org))
        else:
            req.org_id.CopyFrom(identifier_pb2.OrgIdentifier(name=org))
        return await self._select_and_build(req)


class ClientSet:
    def __init__(self, session_cfg: SessionConfig):
        self._console = Console(session_cfg.endpoint, session_cfg.insecure)
        self._session_config = session_cfg
        shared = session_cfg.connect_kwargs()
        self._admin_client = ProjectServiceClient(**shared)
        self._task_service = TaskServiceClient(**shared)
        self._app_service = AppServiceClient(**shared)
        self._run_service = RunServiceClient(**shared)
        self._local_run_service = LocalRunServiceClient(**shared)
        self._log_service = RunLogsServiceClient(**shared)
        self._identity_service = IdentityServiceClient(**shared)
        self._trigger_service = TriggerServiceClient(**shared)
        self._cluster_service = ClusterServiceClient(**shared)
        self._settings_service = SettingsServiceClient(**shared)
        self._secrets_service = ClusterAwareSecretService(
            cluster_service=self._cluster_service,
            session_config=session_cfg,
            default_client=SecretServiceClient(**shared),
        )
        self._dataproxy = ClusterAwareDataProxy(
            cluster_service=self._cluster_service,
            session_config=session_cfg,
            default_client=DataProxyServiceClient(**shared),
        )
        self._image_service = ClusterAwareImageService(
            cluster_service=self._cluster_service,
            session_config=session_cfg,
            default_client=ImageServiceClient(**shared),
        )

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
        return cast(AppService, self._app_service)

    @property
    def run_service(self) -> RunService:
        return cast(RunService, self._run_service)

    @property
    def local_run_service(self) -> LocalRunService:
        """Client for runs orchestrated outside the platform (local runs).

        Local-run RPCs are control-plane only and never route to a dataplane cluster.
        """
        return cast(LocalRunService, self._local_run_service)

    @property
    def dataproxy_service(self) -> DataProxyService:
        """Cluster-aware DataProxy client.

        Each call routes to the cluster selected by ClusterService.SelectCluster
        for the target resource, with per-cluster clients cached.
        """
        return self._dataproxy

    @property
    def image_service(self) -> ImageService:
        """Cluster-aware Image client.

        Each call routes to the cluster selected by ClusterService.SelectCluster
        with OPERATION_GET_IMAGE, with per-cluster clients cached.
        """
        return self._image_service

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
    def cluster_service(self) -> ClusterService:
        return self._cluster_service

    @property
    def settings_service(self) -> SettingsService:
        return self._settings_service

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

from __future__ import annotations

import os

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

    async def close(self, grace: float | None = None):
        return await self._channel.close(grace=grace)

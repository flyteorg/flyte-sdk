from __future__ import annotations

from flyteidl2.actions.actions_service_connect import ActionsServiceClient
from flyteidl2.workflow.queue_service_connect import QueueServiceClient
from flyteidl2.workflow.state_service_connect import StateServiceClient

from flyte.remote._client.auth._session import create_session_config

from ._service_protocol import ActionsService, QueueService, StateService, use_actions_service


class ControllerClient:
    """
    A client for the Controller API.
    """

    def __init__(self, endpoint: str, *, interceptors=(), http_client=None):
        shared = {"address": endpoint, "interceptors": interceptors, "http_client": http_client}
        self._state_service = StateServiceClient(**shared)
        self._queue_service = QueueServiceClient(**shared)
        self._actions_service = ActionsServiceClient(**shared) if use_actions_service() else None

    @classmethod
    async def for_endpoint(cls, endpoint: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session = await create_session_config(endpoint, None, insecure=insecure, **kwargs)
        return cls(session.endpoint, interceptors=session.interceptors, http_client=session.http_client)

    @classmethod
    async def for_api_key(cls, api_key: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session = await create_session_config(None, api_key, insecure=insecure, **kwargs)
        return cls(session.endpoint, interceptors=session.interceptors, http_client=session.http_client)

    @property
    def state_service(self) -> StateService:
        return self._state_service

    @property
    def queue_service(self) -> QueueService:
        return self._queue_service

    @property
    def actions_service(self) -> ActionsService | None:
        return self._actions_service

    async def close(self, grace: float | None = None):
        pass  # no-op for ConnectRPC

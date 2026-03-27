from __future__ import annotations

from flyteidl2.actions.actions_service_connect import ActionsServiceClient
from flyteidl2.workflow.queue_service_connect import QueueServiceClient
from flyteidl2.workflow.state_service_connect import StateServiceClient

from flyte.remote._client.auth._session import SessionConfig, create_session_config

from ._service_protocol import ActionsService, QueueService, StateService, use_actions_service


class ControllerClient:
    """
    A client for the Controller API.
    """

    def __init__(self, session_cfg: SessionConfig):
        shared = session_cfg.connect_kwargs()
        self._state_service = StateServiceClient(**shared)
        self._queue_service = QueueServiceClient(**shared)
        self._actions_service = ActionsServiceClient(**shared) if use_actions_service() else None

    @classmethod
    async def for_endpoint(cls, endpoint: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session_cfg = await create_session_config(endpoint, None, insecure=insecure, **kwargs)
        return cls(session_cfg)

    @classmethod
    async def for_api_key(cls, api_key: str, insecure: bool = False, **kwargs) -> ControllerClient:
        session_cfg = await create_session_config(None, api_key, insecure=insecure, **kwargs)
        return cls(session_cfg)

    @property
    def state_service(self) -> StateService:
        """
        The state service.
        """
        return self._state_service

    @property
    def queue_service(self) -> QueueService:
        """
        The queue service.
        """
        return self._queue_service

    @property
    def actions_service(self) -> ActionsService | None:
        """
        The unified actions service (replaces QueueService + StateService when available).
        """
        return self._actions_service

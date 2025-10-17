from __future__ import annotations

import grpc.aio
from flyteidl2.workflow import queue_service_pb2_grpc, state_service_pb2_grpc

from flyte.remote import create_channel

from ._service_protocol import QueueService, StateService


class ControllerClient:
    """
    A client for the Controller API.
    """

    def __init__(self, channel: grpc.aio.Channel):
        self._channel = channel
        self._state_service = state_service_pb2_grpc.StateServiceStub(channel=channel)
        self._queue_service = queue_service_pb2_grpc.QueueServiceStub(channel=channel)

    @classmethod
    async def for_endpoint(cls, endpoint: str, insecure: bool = False, **kwargs) -> ControllerClient:
        return cls(await create_channel(endpoint, None, insecure=insecure, **kwargs))

    @classmethod
    async def for_api_key(cls, api_key: str, insecure: bool = False, **kwargs) -> ControllerClient:
        return cls(await create_channel(None, api_key, insecure=insecure, **kwargs))

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

    def close(self, grace: float | None = None):
        """
        Close the channel.
        """
        return self._channel.close(grace=grace)

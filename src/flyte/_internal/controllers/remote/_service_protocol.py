from __future__ import annotations

from typing import AsyncIterator, Protocol

from flyteidl2.workflow import queue_service_pb2, state_service_pb2


class StateService(Protocol):
    """
    Interface for the state store client, which stores the history of all subruns.
    """

    async def Watch(
        self, req: state_service_pb2.WatchRequest, **kwargs
    ) -> AsyncIterator[state_service_pb2.WatchResponse]:
        """Watch for subrun updates"""


class QueueService(Protocol):
    """
    Interface for the remote queue service, which is responsible for managing the queue of tasks.
    """

    async def EnqueueAction(
        self,
        req: queue_service_pb2.EnqueueActionRequest,
        **kwargs,
    ) -> queue_service_pb2.EnqueueActionResponse:
        """Enqueue a task"""

    async def AbortQueuedAction(
        self,
        req: queue_service_pb2.AbortQueuedActionRequest,
        **kwargs,
    ) -> queue_service_pb2.AbortQueuedActionResponse:
        """Cancel an enqueued task"""


class ClientSet(Protocol):
    """
    Interface for the remote client set, which is responsible for managing the queue of tasks.
    """

    @property
    def state_service(self: ClientSet) -> StateService:
        """State service"""

    @property
    def queue_service(self: ClientSet) -> QueueService:
        """Queue service"""

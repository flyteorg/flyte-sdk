from __future__ import annotations

import os
from typing import AsyncIterator, Protocol

from flyteidl2.actions import actions_service_pb2
from flyteidl2.workflow import queue_service_pb2, state_service_pb2

ACTIONS_SERVICE_CHECK_ENV_VAR = "_U_USE_ACTIONS"


def use_actions_service() -> bool:
    """Check if the unified ActionsService should be used instead of QueueService + StateService."""
    return os.getenv(ACTIONS_SERVICE_CHECK_ENV_VAR) == "1"


class StateService(Protocol):
    """
    Interface for the state store client, which stores the history of all subruns.
    """

    async def Watch(
        self, req: state_service_pb2.WatchRequest, **kwargs
    ) -> AsyncIterator[state_service_pb2.WatchResponse]:
        """Watch for subrun updates"""


class ActionsService(Protocol):
    """
    Interface for the unified actions service, which replaces both StateService and QueueService.
    """

    async def Enqueue(
        self,
        req: actions_service_pb2.EnqueueRequest,
        **kwargs,
    ) -> actions_service_pb2.EnqueueResponse:
        """Enqueue an action for execution"""

    async def WatchForUpdates(
        self,
        req: actions_service_pb2.WatchForUpdatesRequest,
        **kwargs,
    ) -> AsyncIterator[actions_service_pb2.WatchForUpdatesResponse]:
        """Watch for action state updates"""

    async def Abort(
        self,
        req: actions_service_pb2.AbortRequest,
        **kwargs,
    ) -> actions_service_pb2.AbortResponse:
        """Abort an action"""


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

    @property
    def actions_service(self: ClientSet) -> ActionsService | None:
        """Unified actions service (replaces QueueService + StateService when available)"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator

import grpc.aio
from flyteidl2.common import identifier_pb2, list_pb2

import flyte.errors
from flyte.syncify import syncify

from .._initialize import ensure_client, get_client, get_init_config
from ._common import ToJSONMixin


@dataclass
class Event(ToJSONMixin):
    pb2: ...

    @syncify
    @classmethod
    async def get(
        cls, name: str, /, run_name: str | None = None, task_name: str | None = None, action_name: str | None = None
    ) -> Event | None:
        """
        Retrieve an existing Event by name and scope. The scope is inferred based on the provided parameters.

        :param name: The name of the Event.
        :param run_name: The name of the Run, if the Event scope is "run
        :param task_name: The name of the Task, if the Event scope is "task".
        :param action_name: The name of the Action, if the Event scope is "action".
        :return: An Event instance if found, otherwise None.
        """
        ensure_client()
        cfg = get_init_config()

        if task_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(task_name=task_name)
        elif run_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(run_name=run_name)
        elif action_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(action_name=action_name)
        else:
            raise flyte.errors.EventScopeRequiredError(
                "At least one of run_name, task_name, or action_name must be provided to determine the event scope."
            )

        try:
            resp = await get_client().event_service.GetEvent(
                request=event_service_pb2.GetEventRequest(
                    name=identifier_pb2.EventIdentifier(
                        org=cfg.org,
                        project=cfg.project,
                        domain=cfg.domain,
                        name=name,
                        scope=scope,
                    ),
                )
            )
            return cls(pb2=resp.event)
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise

    @syncify
    @classmethod
    async def listall(
        cls,
        /,
        run_name: str | None = None,
        task_name: str | None = None,
        action_name: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[Event]:
        """
        List all Events within a specific scope. The scope is inferred based on the provided parameters.
        :param run_name: The name of the Run, if the Event scope is "run
        :param task_name: The name of the Task, if the Event scope is "task".
        :param action_name: The name of the Action, if the Event scope is "action".
        :return: An async iterator of Event instances.
        """
        ensure_client()
        cfg = get_init_config()
        if task_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(task_name=task_name)
        elif run_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(run_name=run_name)
        elif action_name is not None:
            scope = identifier_pb2.EventIdentifier.Scope(action_name=action_name)
        else:
            raise flyte.errors.EventScopeRequiredError(
                "At least one of run_name, task_name, or action_name must be provided to determine the event scope."
            )
        token = None
        while True:
            resp = await get_client().event_service.ListEvents(
                request=event_service_pb2.ListEventsRequest(
                    request=list_pb2.ListRequest(
                        limit=limit,
                        token=token,
                    ),
                    scope=scope,
                )
            )
            for event_pb2 in resp.events:
                yield cls(pb2=event_pb2)
            if not resp.token:
                break
            token = resp.token

    @syncify
    async def signal(self, payload: ...) -> None:
        """
        Signal the event with the provided payload.

        :param payload: The payload to signal the event with.
        """
        ensure_client()
        cfg = get_init_config()

        resp = await get_client().event_service.SignalEvent(
            request=event_service_pb2.SignalEventRequest(
                name=identifier_pb2.EventIdentifier(
                    org=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                    name=self.pb2.name,
                    scope=self.pb2.scope,
                ),
                payload=payload,
            )
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

import grpc.aio
from flyteidl2.common import identifier_pb2, list_pb2

try:
    from flyteidl2.workflow import event_service_pb2
except ImportError:
    event_service_pb2 = None  # type: ignore[assignment]

from flyte.syncify import syncify

from .._initialize import ensure_client, get_client, get_init_config
from ._common import ToJSONMixin

# The valid payload types for event signals, matching _event.EventType.
EventPayload = Union[bool, int, float, str]


@dataclass
class Event(ToJSONMixin):
    """
    A remote Event that is registered within an action of a run.

    Events are always scoped to a specific action within a run, identified by
    ``run_name`` and ``action_name``.
    """

    pb2: Any

    @syncify
    @classmethod
    async def get(
        cls,
        name: str,
        /,
        run_name: str,
        action_name: str,
    ) -> Event | None:
        """
        Retrieve an existing Event by name within a specific action of a run.

        :param name: The name of the Event.
        :param run_name: The name of the Run the event belongs to.
        :param action_name: The name of the Action the event belongs to.
        :return: An Event instance if found, otherwise None.
        """
        ensure_client()
        cfg = get_init_config()

        event_id = _make_event_identifier(cfg, name, run_name, action_name)

        try:
            resp = await get_client().event_service.GetEvent(  # type: ignore[attr-defined]
                request=event_service_pb2.GetEventRequest(  # type: ignore[union-attr]
                    event_id=event_id,
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
        run_name: str,
        action_name: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[Event]:
        """
        List all Events for a run, optionally filtered to a specific action.

        :param run_name: The name of the Run to list events for (required).
        :param action_name: Optionally narrow to a specific action within the run.
        :param limit: The maximum number of events to fetch per page.
        :return: An async iterator of Event instances.
        """
        ensure_client()
        cfg = get_init_config()

        run_id = identifier_pb2.RunIdentifier(
            org=cfg.org,
            project=cfg.project,
            domain=cfg.domain,
            name=run_name,
        )

        if action_name is not None:
            action_id = identifier_pb2.ActionIdentifier(run=run_id, name=action_name)
        else:
            action_id = None

        token = None
        while True:
            resp = await get_client().event_service.ListEvents(  # type: ignore[attr-defined]
                request=event_service_pb2.ListEventsRequest(  # type: ignore[union-attr]
                    run_id=run_id,
                    action_id=action_id,
                    request=list_pb2.ListRequest(
                        limit=limit,
                        token=token,
                    ),
                )
            )
            for ev in resp.events:
                yield cls(pb2=ev)
            if not resp.token:
                break
            token = resp.token

    @syncify
    async def signal(self, payload: EventPayload) -> None:
        """
        Signal the event with the provided payload.

        The payload must be one of: ``bool``, ``int``, ``float``, or ``str``.

        :param payload: The value to signal the event with.
        :raises TypeError: If the payload is not a supported type.
        """
        if not isinstance(payload, (bool, int, float, str)):
            raise TypeError(f"payload must be bool, int, float, or str, got {type(payload).__name__}")

        ensure_client()

        await get_client().event_service.SignalEvent(  # type: ignore[attr-defined]
            request=event_service_pb2.SignalEventRequest(  # type: ignore[union-attr]
                event_id=self.pb2.event_id,
                payload=_encode_payload(payload),
            )
        )


def _make_event_identifier(cfg, name: str, run_name: str, action_name: str) -> Any:
    """Build an EventIdentifier from config + names."""
    run_id = identifier_pb2.RunIdentifier(
        org=cfg.org,
        project=cfg.project,
        domain=cfg.domain,
        name=run_name,
    )
    action_id = identifier_pb2.ActionIdentifier(run=run_id, name=action_name)
    return event_service_pb2.EventIdentifier(  # type: ignore[union-attr]
        action_id=action_id,
        name=name,
    )


def _encode_payload(value: EventPayload) -> Any:
    """Encode a Python value into an EventPayload proto message."""
    if isinstance(value, bool):
        return event_service_pb2.EventPayload(bool_value=value)  # type: ignore[union-attr]
    elif isinstance(value, int):
        return event_service_pb2.EventPayload(int_value=value)  # type: ignore[union-attr]
    elif isinstance(value, float):
        return event_service_pb2.EventPayload(float_value=value)  # type: ignore[union-attr]
    elif isinstance(value, str):
        return event_service_pb2.EventPayload(string_value=value)  # type: ignore[union-attr]
    raise TypeError(f"Unsupported payload type: {type(value)}")

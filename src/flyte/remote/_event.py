from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Union

from flyteidl2.common import identifier_pb2, list_pb2, phase_pb2
from flyteidl2.core import types_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2

from flyte.syncify import syncify

from .._initialize import ensure_client, get_client, get_init_config
from ._common import ToJSONMixin

# The valid payload types for event signals, matching _event.EventType.
EventPayload = Union[bool, int, float, str]


@dataclass
class Event(ToJSONMixin):
    """
    A remote Event registered within an action of a run.

    Events pause a run until an external signal is delivered. On the backend an event is
    backed by a *condition action*, so an ``Event`` simply wraps the condition
    :class:`~flyteidl2.workflow.run_definition_pb2.Action` it represents.

    Use :meth:`listall` to discover the events of a run, :meth:`get` to look one up by
    name, and :meth:`signal` to resolve one with a typed payload.
    """

    pb2: run_definition_pb2.Action

    @property
    def name(self) -> str:
        """The event name (the condition action's declared name)."""
        if self.pb2.metadata.HasField("condition"):
            return self.pb2.metadata.condition.name
        return self.pb2.id.name

    @property
    def action_name(self) -> str:
        """The name of the condition action backing this event."""
        return self.pb2.id.name

    @property
    def run_name(self) -> str:
        """The name of the run this event belongs to."""
        return self.pb2.id.run.name

    @property
    def phase(self) -> str:
        """The current phase of the underlying condition action (e.g. ``RUNNING``)."""
        return phase_pb2.ActionPhase.Name(self.pb2.status.phase)

    @property
    def expected_type(self) -> type | None:
        """Python type the condition expects for its payload, derived from
        ``metadata.condition.type`` populated by the backend. Returns ``None`` if the
        underlying action is not a condition or the backend has not yet exposed the
        type (older deployments / older ``flyteidl2`` stubs).
        """
        if not self.pb2.metadata.HasField("condition"):
            return None
        ct = self.pb2.metadata.condition
        try:
            # `type` is a recent ConditionActionMetadata field; older proto stubs
            # don't know about it and `HasField` would raise ValueError.
            if not ct.HasField("type"):
                return None
        except ValueError:
            return None
        return _SIMPLE_TO_PY.get(ct.type.simple)

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
        List all Events for a run, optionally filtered to a specific parent action.

        Events are condition actions, so this lists the run's actions filtered (server
        side) to ``ACTION_TYPE_CONDITION``.

        :param run_name: The name of the Run to list events for (required).
        :param action_name: Optionally narrow to events whose parent is this action.
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
        # Filter to condition actions on the backend rather than client-side.
        condition_filter = list_pb2.Filter(
            function=list_pb2.Filter.Function.EQUAL,
            field="action_type",
            values=[str(int(run_definition_pb2.ACTION_TYPE_CONDITION))],
        )

        token = None
        while True:
            resp = await get_client().run_service.list_actions(
                run_service_pb2.ListActionsRequest(
                    run_id=run_id,
                    request=list_pb2.ListRequest(
                        limit=limit,
                        token=token,
                        filters=[condition_filter],
                    ),
                )
            )
            for action in resp.actions:
                if action_name is not None and action.metadata.parent != action_name:
                    continue
                yield cls(pb2=action)
            if not resp.token:
                break
            token = resp.token

    @syncify
    @classmethod
    async def get(
        cls,
        name: str,
        /,
        run_name: str,
        action_name: str | None = None,
    ) -> Event | None:
        """
        Retrieve an existing Event by name within a run.

        There is no dedicated get-event RPC, so this scans the run's condition actions
        and returns the first whose name matches.

        :param name: The name of the Event.
        :param run_name: The name of the Run the event belongs to.
        :param action_name: Optionally narrow to a specific parent action within the run.
        :return: An Event instance if found, otherwise None.
        """
        async for event in cls.listall.aio(run_name=run_name, action_name=action_name):
            if event.name == name:
                return event
        return None

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

        await get_client().run_service.signal_event(
            run_service_pb2.SignalEventRequest(
                action_id=self.pb2.id,
                parent_action_name=self.pb2.metadata.parent,
                payload=_encode_payload(payload),
            )
        )


def _encode_payload(value: EventPayload) -> Any:
    """Encode a Python value into an EventPayload proto message."""
    # bool must be checked before int, since bool is a subclass of int.
    if isinstance(value, bool):
        return run_service_pb2.EventPayload(bool_value=value)
    elif isinstance(value, int):
        return run_service_pb2.EventPayload(int_value=value)
    elif isinstance(value, float):
        return run_service_pb2.EventPayload(float_value=value)
    elif isinstance(value, str):
        return run_service_pb2.EventPayload(string_value=value)
    raise TypeError(f"Unsupported payload type: {type(value)}")


_SIMPLE_TO_PY: dict[int, type] = {
    types_pb2.BOOLEAN: bool,
    types_pb2.INTEGER: int,
    types_pb2.FLOAT: float,
    types_pb2.STRING: str,
}

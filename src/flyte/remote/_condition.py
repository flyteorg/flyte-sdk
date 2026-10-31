from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, AsyncIterator, Union

import rich.repr
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.core import types_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2

from flyte.syncify import syncify

from .._initialize import ensure_client, get_client, get_init_config
from ._common import ToJSONMixin

# The valid payload types for condition signals, matching _condition.ConditionType.
ConditionPayload = Union[bool, int, float, str]


@dataclass
class Condition(ToJSONMixin):
    """
    A remote Condition registered within an action of a run.

    Conditions pause a run until an external signal is delivered. On the backend a condition is
    backed by a *condition action*, so a `Condition` simply wraps the condition
    `flyteidl2.workflow.run_definition_pb2.Action` it represents.

    Use `Condition.listall` to discover the conditions of a run, `Condition.get` to look one up by
    name, and `Condition.signal` to resolve one with a typed payload.

    What the condition asks for (`prompt`, `prompt_type`, `description`, `timeout`) is not part of
    the action listing; it is only returned by the action-details RPC. Call `Condition.fetch_details`
    to load it, until then those properties are `None`.

    Example:
        ```python
        cond = flyte.remote.Condition.get("approve", run_name="r1").fetch_details()
        print(cond.prompt, cond.expected_type, cond.timeout)
        ```
    """

    pb2: run_definition_pb2.Action
    details_pb2: run_definition_pb2.ActionDetails | None = None

    @property
    def name(self) -> str:
        """The condition name (the condition action's declared name)."""
        if self.pb2.metadata.HasField("condition"):
            return self.pb2.metadata.condition.name
        return self.pb2.id.name

    @property
    def action_name(self) -> str:
        """The name of the condition action backing this condition."""
        return self.pb2.id.name

    @property
    def run_name(self) -> str:
        """The name of the run this condition belongs to."""
        return self.pb2.id.run.name

    @property
    def phase(self) -> str:
        """The current phase of the underlying condition action (e.g. `RUNNING`)."""
        from flyte._utils.helpers import action_phase_name

        return action_phase_name(self.pb2.status.phase)

    @property
    def prompt(self) -> str | None:
        """The prompt text the condition shows whoever signals it, or `None` if details are not loaded."""
        spec = self._spec
        if spec is None or not spec.prompt:
            return None
        return spec.prompt

    @property
    def prompt_type(self) -> str | None:
        """How `prompt` should be rendered: `"text"` or `"markdown"`, or `None` if unknown."""
        spec = self._spec
        if spec is None:
            return None
        return _PROMPT_TYPE_TO_STR.get(spec.prompt_type)

    @property
    def description(self) -> str | None:
        """The condition's description, or `None` if it has none or details are not loaded."""
        spec = self._spec
        if spec is None or not spec.description:
            return None
        return spec.description

    @property
    def timeout(self) -> timedelta | None:
        """How long the condition waits for a signal before timing out, or `None` if it waits forever
        or details are not loaded."""
        spec = self._spec
        if spec is None or not spec.HasField("timeout"):
            return None
        return spec.timeout.ToTimedelta()

    @property
    def _spec(self) -> run_definition_pb2.ConditionAction | None:
        if self.details_pb2 is None or not self.details_pb2.HasField("condition"):
            return None
        return self.details_pb2.condition

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "action", self.action_name
        yield "run", self.run_name
        yield "phase", self.phase
        if (et := self.expected_type) is not None:
            yield "type", et.__name__
        if (prompt := self.prompt) is not None:
            yield "prompt", prompt
        if (timeout := self.timeout) is not None:
            yield "timeout", timeout
        yield "parent", self.pb2.metadata.parent

    @property
    def expected_type(self) -> type | None:
        """Python type the condition expects for its payload, derived from
        `metadata.condition.type` populated by the backend, falling back to the loaded
        details (see `fetch_details`). Returns `None` if the underlying action is not a
        condition or the backend has not exposed the type (older deployments / older
        `flyteidl2` stubs).
        """
        from_metadata = self._metadata_expected_type()
        if from_metadata is not None:
            return from_metadata
        spec = self._spec
        return _data_type_from_literal_type(spec.type) if spec is not None else None

    def _metadata_expected_type(self) -> type | None:
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
    async def fetch_details(self) -> Condition:
        """
        Load what the condition is asking for: its prompt, description and timeout.

        This is one action-details RPC for the condition action. The current object is left
        unchanged; a new `Condition` carrying the details is returned.

        Returns:
            A Condition whose `prompt`, `prompt_type`, `description` and `timeout` are populated.
        """
        ensure_client()
        resp = await get_client().run_service.get_action_details(
            run_service_pb2.GetActionDetailsRequest(action_id=self.pb2.id)
        )
        return replace(self, details_pb2=resp.details)

    @syncify
    @classmethod
    async def listall(
        cls,
        /,
        run_name: str,
        action_name: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[Condition]:
        """
        List all Conditions for a run, optionally filtered to a specific parent action.

        Conditions are condition actions, so this lists the run's actions filtered (server
        side) to `ACTION_TYPE_CONDITION`.

        Args:
            run_name: The name of the Run to list conditions for (required).
            action_name: Optionally narrow to conditions whose parent is this action.
            limit: The maximum number of conditions to fetch per page.

        Returns:
            An async iterator of Condition instances.
        """
        ensure_client()
        cfg = get_init_config()

        run_id = identifier_pb2.RunIdentifier(
            org=cfg.org,
            project=cfg.project,
            domain=cfg.domain,
            name=run_name,
        )
        # Filter to condition actions (and optionally to one parent) on the backend rather than client-side.
        filters = [
            list_pb2.Filter(
                function=list_pb2.Filter.Function.EQUAL,
                field="action_type",
                values=[str(int(run_definition_pb2.ACTION_TYPE_CONDITION))],
            )
        ]
        if action_name is not None:
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.Function.EQUAL,
                    field="parent_name",
                    values=[action_name],
                )
            )

        token = None
        while True:
            resp = await get_client().run_service.list_actions(
                run_service_pb2.ListActionsRequest(
                    run_id=run_id,
                    request=list_pb2.ListRequest(
                        limit=limit,
                        token=token,
                        filters=filters,
                    ),
                )
            )
            for action in resp.actions:
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
    ) -> Condition | None:
        """
        Retrieve an existing Condition by name within a run.

        There is no dedicated get-condition RPC, so this scans the run's condition actions
        and returns the first whose name matches.

        Args:
            name: The name of the Condition.
            run_name: The name of the Run the condition belongs to.
            action_name: Optionally narrow to a specific parent action within the run.

        Returns:
            A Condition instance if found, otherwise None.
        """
        async for condition in cls.listall.aio(run_name=run_name, action_name=action_name):
            if condition.name == name:
                return condition
        return None

    @syncify
    async def signal(self, payload: ConditionPayload) -> None:
        """
        Signal the condition with the provided payload.

        The payload must be one of: `bool`, `int`, `float`, or `str`.

        Args:
            payload: The value to signal the condition with.

        Raises:
            TypeError: If the payload is not a supported type.
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


def _encode_payload(value: ConditionPayload) -> Any:
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


_PROMPT_TYPE_TO_STR: dict[int, str] = {
    run_definition_pb2.CONDITION_PROMPT_TYPE_TEXT: "text",
    run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN: "markdown",
}

_SIMPLE_TO_PY: dict[int, type] = {
    types_pb2.BOOLEAN: bool,
    types_pb2.INTEGER: int,
    types_pb2.FLOAT: float,
    types_pb2.STRING: str,
}


def _data_type_from_literal_type(literal_type: types_pb2.LiteralType | None) -> type | None:
    """Map a `LiteralType` proto to a Python condition payload type."""
    if literal_type is None:
        return None
    try:
        if literal_type.HasField("simple"):
            return _SIMPLE_TO_PY.get(literal_type.simple)
    except ValueError:
        pass
    simple = getattr(literal_type, "simple", None)
    if simple is not None:
        return _SIMPLE_TO_PY.get(simple)
    return None


def resolve_condition_expected_type(
    action_pb2: run_definition_pb2.Action,
    *,
    details_pb2: run_definition_pb2.ActionDetails | None = None,
) -> type | None:
    """Return the Python payload type for a condition action.

    Checks `metadata.condition.type` first, then `ActionDetails.condition.type`
    when details are available (the backend often only populates the latter).
    """
    return Condition(pb2=action_pb2, details_pb2=details_pb2).expected_type


_BOOL_TRUE = {"true", "1", "yes", "y", "t"}
_BOOL_FALSE = {"false", "0", "no", "n", "f"}


def coerce_condition_payload(value: Any, expected: type) -> ConditionPayload:
    """Coerce *value* to the condition's declared payload type."""
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in _BOOL_TRUE:
                return True
            if v in _BOOL_FALSE:
                return False
            raise ValueError(f"could not parse {value!r} as bool")
        raise TypeError(f"expected bool payload, got {type(value).__name__}")
    if expected is int:
        if isinstance(value, bool):
            raise TypeError("expected int payload, got bool")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError as e:
                raise ValueError(f"could not parse {value!r} as int") from e
        raise TypeError(f"expected int payload, got {type(value).__name__}")
    if expected is float:
        if isinstance(value, bool):
            raise TypeError("expected float payload, got bool")
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as e:
                raise ValueError(f"could not parse {value!r} as float") from e
        raise TypeError(f"expected float payload, got {type(value).__name__}")
    if expected is str:
        return str(value)
    raise TypeError(f"unsupported expected condition type {expected!r}")

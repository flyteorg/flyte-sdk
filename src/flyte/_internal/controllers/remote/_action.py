from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import execution_pb2, interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2, task_definition_pb2
from flyteidl2.workflow import (
    run_definition_pb2,
    state_service_pb2,
)
from google.protobuf import timestamp_pb2

from flyte.models import GroupData

ActionType = Literal["task", "trace", "condition"]


@dataclass
class Action:
    """
    Coroutine safe, as we never do await operations in any method.
    Holds the inmemory state of a task. It is combined representation of local and remote states.
    """

    action_id: identifier_pb2.ActionIdentifier
    parent_action_name: str
    type: ActionType = "task"  # type of action, task or trace
    friendly_name: str | None = None
    group: GroupData | None = None
    task: task_definition_pb2.TaskSpec | None = None
    trace: run_definition_pb2.TraceAction | None = None
    condition: run_definition_pb2.ConditionAction | None = None
    inputs_uri: str | None = None
    run_output_base: str | None = None
    realized_outputs_uri: str | None = None
    err: execution_pb2.ExecutionError | None = None
    phase: phase_pb2.ActionPhase | None = None
    started: bool = False
    retries: int = 0
    queue: Optional[str] = None  # The queue to which this action was submitted.
    client_err: Exception | None = None  # This error is set when something goes wrong in the controller.
    cache_key: str | None = None  # None means no caching, otherwise it is the version of the cache.
    condition_output: literals_pb2.Literal | None = None  # Output Literal for condition actions (set from ActionUpdate)

    @property
    def name(self) -> str:
        return self.action_id.name

    @property
    def run_name(self) -> str:
        return self.action_id.run.name

    def is_terminal(self) -> bool:
        """Check if resource has reached terminal state"""
        if self.phase is None:
            return False
        return self.phase in [
            phase_pb2.ACTION_PHASE_FAILED,
            phase_pb2.ACTION_PHASE_SUCCEEDED,
            phase_pb2.ACTION_PHASE_ABORTED,
            phase_pb2.ACTION_PHASE_TIMED_OUT,
        ]

    def increment_retries(self):
        self.retries += 1

    def is_started(self) -> bool:
        """Check if resource has been started."""
        return self.started

    def mark_started(self):
        self.started = True
        self.task = None

    def mark_cancelled(self):
        self.mark_started()
        self.phase = phase_pb2.ACTION_PHASE_ABORTED

    def merge_state(self, obj: state_service_pb2.ActionUpdate):
        """
        This method is invoked when the watch API sends an update about the state of the action. We need to merge
        the state of the action with the current state of the action. It is possible that we have no phase information
        prior to this.
        :param obj:
        :return:
        """
        if self.phase != obj.phase:
            self.phase = obj.phase
            self.err = obj.error if obj.HasField("error") else None
        self.realized_outputs_uri = obj.output_uri
        # For condition actions, the backend may include the output Literal directly
        # in the ActionUpdate instead of an output_uri.
        # TODO: Uncomment when the ActionUpdate proto adds the `output` field:
        # if self.type == "condition" and obj.HasField("output"):
        #     self.condition_output = obj.output
        self.started = True

    def merge_in_action_from_submit(self, action: Action):
        """
        This method is invoked when parent_action submits an action that was observed previously observed from the
         watch. We need to merge in the contents of the action, while preserving the observed phase.

        :param action: The submitted action
        """
        self.run_output_base = action.run_output_base
        self.inputs_uri = action.inputs_uri
        self.group = action.group
        self.friendly_name = action.friendly_name
        if not self.started:
            self.task = action.task

        self.cache_key = action.cache_key

    def set_client_error(self, exc: Exception):
        self.client_err = exc

    def has_error(self) -> bool:
        return self.client_err is not None or self.err is not None

    @staticmethod
    def literal_to_python(literal: literals_pb2.Literal, expected_type: builtins.type) -> object:
        """Convert a flyteidl Literal (scalar/primitive) to a Python value.

        The ``expected_type`` must be one of ``bool``, ``int``, ``float``, or ``str``.

        Returns the Python-native value (``True``/``False`` for bool, etc.).
        """
        primitive = literal.scalar.primitive
        if expected_type is bool:
            return bool(primitive.boolean)
        if expected_type is int:
            return int(primitive.integer)
        if expected_type is float:
            return float(primitive.float_value)
        if expected_type is str:
            return str(primitive.string_value)
        raise TypeError(f"Unsupported expected_type {expected_type}")

    @classmethod
    def from_task(
        cls,
        parent_action_name: str,
        sub_action_id: identifier_pb2.ActionIdentifier,
        group_data: GroupData | None,
        task_spec: task_definition_pb2.TaskSpec,
        inputs_uri: str,
        run_output_base: str,
        cache_key: str | None = None,
        queue: Optional[str] = None,
    ) -> Action:
        return cls(
            action_id=sub_action_id,
            parent_action_name=parent_action_name,
            friendly_name=task_spec.task_template.id.name,
            group=group_data,
            task=task_spec,
            inputs_uri=inputs_uri,
            run_output_base=run_output_base,
            cache_key=cache_key,
            queue=queue,
        )

    @classmethod
    def from_state(cls, parent_action_name: str, obj: state_service_pb2.ActionUpdate) -> Action:
        """
        This creates a new action, from the watch api. This is possible in the case of a recovery, where the
        state service knows about future actions and sends this information to the informer. We may not have
        encountered the "task" itself yet, but we know about the action id and the state of the action.

        :param parent_action_name:
        :param obj:
        :return:
        """
        from flyte._logging import logger

        logger.debug(f"In Action from_state {obj.action_id} {obj.phase} {obj.output_uri}")
        # For condition actions, the backend may include the output Literal directly.
        # TODO: Uncomment when the ActionUpdate proto adds the `output` field:
        # condition_output = obj.output if obj.HasField("output") else None
        condition_output = None
        return cls(
            action_id=obj.action_id,
            parent_action_name=parent_action_name,
            phase=obj.phase,
            started=True,
            err=obj.error if obj.HasField("error") else None,
            realized_outputs_uri=obj.output_uri,
            condition_output=condition_output,
        )

    @classmethod
    def from_trace(
        cls,
        parent_action_name: str,
        action_id: identifier_pb2.ActionIdentifier,
        friendly_name: str,
        group_data: GroupData | None,
        inputs_uri: str,
        outputs_uri: str,
        start_time: float,  # Unix timestamp in seconds with fractional seconds
        end_time: float,  # Unix timestamp in seconds with fractional seconds
        run_output_base: str,
        report_uri: str | None = None,
        typed_interface: interface_pb2.TypedInterface | None = None,
    ) -> Action:
        """
        This creates a new action for tracing purposes. It is used to track the execution of a trace.
        """
        st = timestamp_pb2.Timestamp()
        st.FromSeconds(int(start_time))
        st.nanos = int((start_time % 1) * 1e9)

        et = timestamp_pb2.Timestamp()
        et.FromSeconds(int(end_time))
        et.nanos = int((end_time % 1) * 1e9)

        spec = task_definition_pb2.TraceSpec(interface=typed_interface) if typed_interface else None

        return cls(
            action_id=action_id,
            parent_action_name=parent_action_name,
            type="trace",
            friendly_name=friendly_name,
            group=group_data,
            inputs_uri=inputs_uri,
            realized_outputs_uri=outputs_uri,
            phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
            run_output_base=run_output_base,
            trace=run_definition_pb2.TraceAction(
                name=friendly_name,
                phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
                start_time=st,
                end_time=et,
                outputs=common_pb2.OutputReferences(
                    output_uri=outputs_uri,
                    report_uri=report_uri,
                ),
                spec=spec,
            ),
        )

    # Mapping from Python types to flyteidl SimpleType enum values (class var, not a dataclass field)
    _DATA_TYPE_TO_SIMPLE: ClassVar[dict[builtins.type, int]] = {
        bool: types_pb2.BOOLEAN,
        int: types_pb2.INTEGER,
        float: types_pb2.FLOAT,
        str: types_pb2.STRING,
    }

    @classmethod
    def from_condition(
        cls,
        parent_action_name: str,
        action_id: identifier_pb2.ActionIdentifier,
        event_name: str,
        prompt: str,
        data_type: builtins.type,
        run_output_base: str,
        group_data: GroupData | None = None,
        description: str = "",
        # TODO: proto does not yet have these fields — will be added separately
        # prompt_type: str = "text",
        # timeout: float | None = None,
        # webhook_url: str | None = None,
        # webhook_payload: dict | None = None,
    ) -> Action:
        """Create a condition action for an event."""
        simple_type = cls._DATA_TYPE_TO_SIMPLE.get(data_type)
        if simple_type is None:
            raise TypeError(f"Unsupported event data_type {data_type}")

        literal_type = types_pb2.LiteralType(simple=simple_type)

        return cls(
            action_id=action_id,
            parent_action_name=parent_action_name,
            type="condition",
            friendly_name=event_name,
            group=group_data,
            run_output_base=run_output_base,
            condition=run_definition_pb2.ConditionAction(
                name=event_name,
                run_id=action_id.run.name,
                action_id=action_id.name,
                type=literal_type,
                prompt=prompt,
                description=description,
            ),
        )

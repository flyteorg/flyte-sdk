from flyte._protos.common import identifier_pb2 as _identifier_pb2
from flyteidl.core import types_pb2 as _types_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import wrappers_pb2 as _wrappers_pb2
from flyte._protos.validate.validate import validate_pb2 as _validate_pb2
from flyte._protos.workflow import run_definition_pb2 as _run_definition_pb2
from flyte._protos.workflow import task_definition_pb2 as _task_definition_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WorkerIdentifier(_message.Message):
    __slots__ = ["organization", "cluster", "name"]
    ORGANIZATION_FIELD_NUMBER: _ClassVar[int]
    CLUSTER_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    organization: str
    cluster: str
    name: str
    def __init__(self, organization: _Optional[str] = ..., cluster: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class EnqueueActionRequest(_message.Message):
    __slots__ = ["action_id", "parent_action_name", "run_spec", "input_uri", "run_output_base", "group", "subject", "task", "trace", "condition"]
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ACTION_NAME_FIELD_NUMBER: _ClassVar[int]
    RUN_SPEC_FIELD_NUMBER: _ClassVar[int]
    INPUT_URI_FIELD_NUMBER: _ClassVar[int]
    RUN_OUTPUT_BASE_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    TRACE_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    action_id: _identifier_pb2.ActionIdentifier
    parent_action_name: str
    run_spec: _run_definition_pb2.RunSpec
    input_uri: str
    run_output_base: str
    group: str
    subject: str
    task: TaskAction
    trace: TraceAction
    condition: ConditionAction
    def __init__(self, action_id: _Optional[_Union[_identifier_pb2.ActionIdentifier, _Mapping]] = ..., parent_action_name: _Optional[str] = ..., run_spec: _Optional[_Union[_run_definition_pb2.RunSpec, _Mapping]] = ..., input_uri: _Optional[str] = ..., run_output_base: _Optional[str] = ..., group: _Optional[str] = ..., subject: _Optional[str] = ..., task: _Optional[_Union[TaskAction, _Mapping]] = ..., trace: _Optional[_Union[TraceAction, _Mapping]] = ..., condition: _Optional[_Union[ConditionAction, _Mapping]] = ...) -> None: ...

class TaskAction(_message.Message):
    __slots__ = ["id", "spec", "cache_key"]
    ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    CACHE_KEY_FIELD_NUMBER: _ClassVar[int]
    id: _task_definition_pb2.TaskIdentifier
    spec: _task_definition_pb2.TaskSpec
    cache_key: _wrappers_pb2.StringValue
    def __init__(self, id: _Optional[_Union[_task_definition_pb2.TaskIdentifier, _Mapping]] = ..., spec: _Optional[_Union[_task_definition_pb2.TaskSpec, _Mapping]] = ..., cache_key: _Optional[_Union[_wrappers_pb2.StringValue, _Mapping]] = ...) -> None: ...

class TraceAction(_message.Message):
    __slots__ = ["name", "phase", "start_time", "end_time", "outputs", "spec"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    name: str
    phase: _run_definition_pb2.Phase
    start_time: _timestamp_pb2.Timestamp
    end_time: _timestamp_pb2.Timestamp
    outputs: _run_definition_pb2.OutputReferences
    spec: _task_definition_pb2.TaskSpec
    def __init__(self, name: _Optional[str] = ..., phase: _Optional[_Union[_run_definition_pb2.Phase, str]] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., end_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., outputs: _Optional[_Union[_run_definition_pb2.OutputReferences, _Mapping]] = ..., spec: _Optional[_Union[_task_definition_pb2.TaskSpec, _Mapping]] = ...) -> None: ...

class ConditionAction(_message.Message):
    __slots__ = ["name", "run_id", "action_id", "type", "prompt", "description"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    run_id: str
    action_id: str
    type: _types_pb2.LiteralType
    prompt: str
    description: str
    def __init__(self, name: _Optional[str] = ..., run_id: _Optional[str] = ..., action_id: _Optional[str] = ..., type: _Optional[_Union[_types_pb2.LiteralType, _Mapping]] = ..., prompt: _Optional[str] = ..., description: _Optional[str] = ..., **kwargs) -> None: ...

class EnqueueActionResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class AbortQueuedRunRequest(_message.Message):
    __slots__ = ["run_id"]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: _identifier_pb2.RunIdentifier
    def __init__(self, run_id: _Optional[_Union[_identifier_pb2.RunIdentifier, _Mapping]] = ...) -> None: ...

class AbortQueuedRunResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ["worker_id", "active_action_ids", "terminal_action_ids", "aborted_action_ids", "available_capacity"]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_ACTION_IDS_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_ACTION_IDS_FIELD_NUMBER: _ClassVar[int]
    ABORTED_ACTION_IDS_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    worker_id: WorkerIdentifier
    active_action_ids: _containers.RepeatedCompositeFieldContainer[_identifier_pb2.ActionIdentifier]
    terminal_action_ids: _containers.RepeatedCompositeFieldContainer[_identifier_pb2.ActionIdentifier]
    aborted_action_ids: _containers.RepeatedCompositeFieldContainer[_identifier_pb2.ActionIdentifier]
    available_capacity: int
    def __init__(self, worker_id: _Optional[_Union[WorkerIdentifier, _Mapping]] = ..., active_action_ids: _Optional[_Iterable[_Union[_identifier_pb2.ActionIdentifier, _Mapping]]] = ..., terminal_action_ids: _Optional[_Iterable[_Union[_identifier_pb2.ActionIdentifier, _Mapping]]] = ..., aborted_action_ids: _Optional[_Iterable[_Union[_identifier_pb2.ActionIdentifier, _Mapping]]] = ..., available_capacity: _Optional[int] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ["new_leases", "aborted_leases", "finalized_action_ids"]
    NEW_LEASES_FIELD_NUMBER: _ClassVar[int]
    ABORTED_LEASES_FIELD_NUMBER: _ClassVar[int]
    FINALIZED_ACTION_IDS_FIELD_NUMBER: _ClassVar[int]
    new_leases: _containers.RepeatedCompositeFieldContainer[Lease]
    aborted_leases: _containers.RepeatedCompositeFieldContainer[Lease]
    finalized_action_ids: _containers.RepeatedCompositeFieldContainer[_identifier_pb2.ActionIdentifier]
    def __init__(self, new_leases: _Optional[_Iterable[_Union[Lease, _Mapping]]] = ..., aborted_leases: _Optional[_Iterable[_Union[Lease, _Mapping]]] = ..., finalized_action_ids: _Optional[_Iterable[_Union[_identifier_pb2.ActionIdentifier, _Mapping]]] = ...) -> None: ...

class StreamLeasesRequest(_message.Message):
    __slots__ = ["worker_id"]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    worker_id: WorkerIdentifier
    def __init__(self, worker_id: _Optional[_Union[WorkerIdentifier, _Mapping]] = ...) -> None: ...

class StreamLeasesResponse(_message.Message):
    __slots__ = ["leases"]
    LEASES_FIELD_NUMBER: _ClassVar[int]
    leases: _containers.RepeatedCompositeFieldContainer[Lease]
    def __init__(self, leases: _Optional[_Iterable[_Union[Lease, _Mapping]]] = ...) -> None: ...

class Lease(_message.Message):
    __slots__ = ["action_id", "parent_action_name", "run_spec", "input_uri", "run_output_base", "task", "condition", "trace", "group", "subject", "host"]
    ACTION_ID_FIELD_NUMBER: _ClassVar[int]
    PARENT_ACTION_NAME_FIELD_NUMBER: _ClassVar[int]
    RUN_SPEC_FIELD_NUMBER: _ClassVar[int]
    INPUT_URI_FIELD_NUMBER: _ClassVar[int]
    RUN_OUTPUT_BASE_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    TRACE_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    action_id: _identifier_pb2.ActionIdentifier
    parent_action_name: str
    run_spec: _run_definition_pb2.RunSpec
    input_uri: str
    run_output_base: str
    task: TaskAction
    condition: ConditionAction
    trace: TraceAction
    group: str
    subject: str
    host: str
    def __init__(self, action_id: _Optional[_Union[_identifier_pb2.ActionIdentifier, _Mapping]] = ..., parent_action_name: _Optional[str] = ..., run_spec: _Optional[_Union[_run_definition_pb2.RunSpec, _Mapping]] = ..., input_uri: _Optional[str] = ..., run_output_base: _Optional[str] = ..., task: _Optional[_Union[TaskAction, _Mapping]] = ..., condition: _Optional[_Union[ConditionAction, _Mapping]] = ..., trace: _Optional[_Union[TraceAction, _Mapping]] = ..., group: _Optional[str] = ..., subject: _Optional[str] = ..., host: _Optional[str] = ...) -> None: ...

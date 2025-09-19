from flyte._protos.common import identifier_pb2 as _identifier_pb2
from flyte._protos.common import identity_pb2 as _identity_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from flyte._protos.validate.validate import validate_pb2 as _validate_pb2
from flyte._protos.workflow import run_definition_pb2 as _run_definition_pb2
from flyte._protos.workflow import task_definition_pb2 as _task_definition_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FixedRateUnit(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    FIXED_RATE_UNIT_UNSPECIFIED: _ClassVar[FixedRateUnit]
    FIXED_RATE_UNIT_MINUTE: _ClassVar[FixedRateUnit]
    FIXED_RATE_UNIT_HOUR: _ClassVar[FixedRateUnit]
    FIXED_RATE_UNIT_DAY: _ClassVar[FixedRateUnit]
FIXED_RATE_UNIT_UNSPECIFIED: FixedRateUnit
FIXED_RATE_UNIT_MINUTE: FixedRateUnit
FIXED_RATE_UNIT_HOUR: FixedRateUnit
FIXED_RATE_UNIT_DAY: FixedRateUnit

class TriggerMetadata(_message.Message):
    __slots__ = ["deployed_by", "updated_by"]
    DEPLOYED_BY_FIELD_NUMBER: _ClassVar[int]
    UPDATED_BY_FIELD_NUMBER: _ClassVar[int]
    deployed_by: _identity_pb2.EnrichedIdentity
    updated_by: _identity_pb2.EnrichedIdentity
    def __init__(self, deployed_by: _Optional[_Union[_identity_pb2.EnrichedIdentity, _Mapping]] = ..., updated_by: _Optional[_Union[_identity_pb2.EnrichedIdentity, _Mapping]] = ...) -> None: ...

class FixedRate(_message.Message):
    __slots__ = ["value", "unit", "start_time"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    value: int
    unit: FixedRateUnit
    start_time: _timestamp_pb2.Timestamp
    def __init__(self, value: _Optional[int] = ..., unit: _Optional[_Union[FixedRateUnit, str]] = ..., start_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Schedule(_message.Message):
    __slots__ = ["rate", "cron_expression", "kickoff_time_input_arg"]
    RATE_FIELD_NUMBER: _ClassVar[int]
    CRON_EXPRESSION_FIELD_NUMBER: _ClassVar[int]
    KICKOFF_TIME_INPUT_ARG_FIELD_NUMBER: _ClassVar[int]
    rate: FixedRate
    cron_expression: str
    kickoff_time_input_arg: str
    def __init__(self, rate: _Optional[_Union[FixedRate, _Mapping]] = ..., cron_expression: _Optional[str] = ..., kickoff_time_input_arg: _Optional[str] = ...) -> None: ...

class TriggerSpec(_message.Message):
    __slots__ = ["task_id", "inputs", "run_spec", "active", "schedule"]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    RUN_SPEC_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    task_id: _task_definition_pb2.TaskIdentifier
    inputs: _run_definition_pb2.Inputs
    run_spec: _run_definition_pb2.RunSpec
    active: bool
    schedule: Schedule
    def __init__(self, task_id: _Optional[_Union[_task_definition_pb2.TaskIdentifier, _Mapping]] = ..., inputs: _Optional[_Union[_run_definition_pb2.Inputs, _Mapping]] = ..., run_spec: _Optional[_Union[_run_definition_pb2.RunSpec, _Mapping]] = ..., active: bool = ..., schedule: _Optional[_Union[Schedule, _Mapping]] = ...) -> None: ...

class TriggerStatus(_message.Message):
    __slots__ = ["deployed_at", "updated_at", "triggered_at"]
    DEPLOYED_AT_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    TRIGGERED_AT_FIELD_NUMBER: _ClassVar[int]
    deployed_at: _timestamp_pb2.Timestamp
    updated_at: _timestamp_pb2.Timestamp
    triggered_at: _timestamp_pb2.Timestamp
    def __init__(self, deployed_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., triggered_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class Trigger(_message.Message):
    __slots__ = ["id", "meta", "status", "task_id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    id: _identifier_pb2.TriggerIdentifier
    meta: TriggerMetadata
    status: TriggerStatus
    task_id: _task_definition_pb2.TaskIdentifier
    def __init__(self, id: _Optional[_Union[_identifier_pb2.TriggerIdentifier, _Mapping]] = ..., meta: _Optional[_Union[TriggerMetadata, _Mapping]] = ..., status: _Optional[_Union[TriggerStatus, _Mapping]] = ..., task_id: _Optional[_Union[_task_definition_pb2.TaskIdentifier, _Mapping]] = ...) -> None: ...

class TriggerDetails(_message.Message):
    __slots__ = ["id", "meta", "spec", "status"]
    ID_FIELD_NUMBER: _ClassVar[int]
    META_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    id: _identifier_pb2.TriggerIdentifier
    meta: TriggerMetadata
    spec: TriggerSpec
    status: TriggerStatus
    def __init__(self, id: _Optional[_Union[_identifier_pb2.TriggerIdentifier, _Mapping]] = ..., meta: _Optional[_Union[TriggerMetadata, _Mapping]] = ..., spec: _Optional[_Union[TriggerSpec, _Mapping]] = ..., status: _Optional[_Union[TriggerStatus, _Mapping]] = ...) -> None: ...

from typing import Union

from flyteidl.core import literals_pb2
from google.protobuf import wrappers_pb2

from flyte import Cron, FixedRate, Trigger, TriggerTime
from flyteidl2.task import common_pb2, run_pb2
from flyteidl2.trigger import trigger_definition_pb2


def _to_schedule(m: Union[Cron, FixedRate], kickoff_arg_name: str | None = None) -> common_pb2.Schedule:
    if isinstance(m, Cron):
        return common_pb2.Schedule(
            cron_expression=m.expression,
            kickoff_time_input_arg=kickoff_arg_name,
        )
    elif isinstance(m, FixedRate):
        start_time = wrappers_pb2.StringValue(m.start_time.isoformat()) if m.start_time is not None else None

        return common_pb2.Schedule(
            rate=common_pb2.FixedRate(
                value=m.interval_minutes,
                unit=common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE,
                start_time=start_time,
            ),
            kickoff_time_input_arg=kickoff_arg_name,
        )


def to_task_trigger(t: Trigger) -> trigger_definition_pb2.TriggerSpec:
    env = None
    if t.env_vars:
        env = run_pb2.Envs([literals_pb2.KeyValuePair(key=k, value=v) for k, v in t.env_vars.items()])

    labels = run_pb2.Labels(values=t.labels) if t.labels else None

    annotations = run_pb2.Annotations(values=t.annotations) if t.annotations else None

    run_spec = run_pb2.RunSpec(
        overwrite_cache=t.overwrite_cache,
        envs=env,
        interruptible=wrappers_pb2.BoolValue(t.interruptible) if t.interruptible is not None else None,
        cluster=t.queue,
        labels=labels,
        annotations=annotations,
    )

    kickoff_arg_name = None
    if t.inputs:
        for k, v in t.inputs.items():
            if v is TriggerTime:
                kickoff_arg_name = k
                # Continue to process other inputs, don't break
                continue

    automation = _to_schedule(
        t.automation,
        kickoff_arg_name=kickoff_arg_name,
    )

    return trigger_definition_pb2.TriggerSpec(
        active=t.auto_activate,
        run_spec=run_spec,
        inputs=None,  # No inputs for now
        automation_spec=common_pb2.TriggerAutomationSpec(
            type=common_pb2.TriggerAutomationSpec.Type.TYPE_SCHEDULE,
            schedule=automation,
        ),
    )

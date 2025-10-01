import asyncio
from typing import Union

from flyteidl.core import interface_pb2, literals_pb2
from google.protobuf import wrappers_pb2

import flyte.types
from flyte import Cron, FixedRate, Trigger, TriggerTime
from flyte._protos.workflow import common_pb2, run_definition_pb2, trigger_definition_pb2


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


async def to_task_trigger(
    t: Trigger,
    task_name: str,
    task_inputs: interface_pb2.VariableMap,
    task_default_inputs: list[common_pb2.NamedParameter],
) -> trigger_definition_pb2.TaskTrigger:
    """
    Converts a Trigger object to a TaskTrigger protobuf object.
    Args:
        t:
        task_name:
        task_inputs:
        task_default_inputs:
    Returns:

    """
    env = None
    if t.env_vars:
        env = run_definition_pb2.Envs([literals_pb2.KeyValuePair(key=k, value=v) for k, v in t.env_vars.items()])

    labels = run_definition_pb2.Labels(values=t.labels) if t.labels else None

    annotations = run_definition_pb2.Annotations(values=t.annotations) if t.annotations else None

    run_spec = run_definition_pb2.RunSpec(
        overwrite_cache=t.overwrite_cache,
        envs=env,
        interruptible=wrappers_pb2.BoolValue(t.interruptible) if t.interruptible is not None else None,
        cluster=t.queue,
        labels=labels,
        annotations=annotations,
    )

    kickoff_arg_name = None
    default_inputs = {}
    if t.inputs:
        for k, v in t.inputs.items():
            if v is TriggerTime:
                kickoff_arg_name = k
                break
            else:
                default_inputs[k] = v

    # assert that default_inputs and the kickoff_arg_name are infact in the task inputs
    if kickoff_arg_name is not None and kickoff_arg_name not in task_inputs.variables:
        raise ValueError(
            f"For a scheduled trigger, the TriggerTime input '{kickoff_arg_name}' "
            f"must be an input to the task, but not found in task {task_name}. "
            f"Available inputs: {list(task_inputs.variables.keys())}"
        )

    keys = []
    literal_coros = []
    for k, v in default_inputs.items():
        if k not in task_inputs.variables:
            raise ValueError(
                f"Trigger default input '{k}' must be an input to the task, but not found in task {task_name}. "
                f"Available inputs: {list(task_inputs.variables.keys())}"
            )
        else:
            literal_coros.append(flyte.types.TypeEngine.to_literal(v, type(v), task_inputs.variables[k].type))
            keys.append(k)

    final_literals: list[literals_pb2.Literal] = await asyncio.gather(*literal_coros)

    for p in task_default_inputs or []:
        if p.name not in keys:
            keys.append(p.name)
            final_literals.append(p.parameter.default)

    literals: list[run_definition_pb2.NamedLiteral] = []
    for k, lit in zip(keys, final_literals):
        literals.append(
            run_definition_pb2.NamedLiteral(
                name=k,
                value=lit,
            )
        )

    automation = _to_schedule(
        t.automation,
        kickoff_arg_name=kickoff_arg_name,
    )

    return trigger_definition_pb2.TaskTrigger(
        name=t.name,
        spec=trigger_definition_pb2.TaskTriggerSpec(
            active=t.auto_activate,
            run_spec=run_spec,
            inputs=run_definition_pb2.Inputs(literals=literals),
        ),
        automation_spec=common_pb2.TriggerAutomationSpec(
            type=common_pb2.TriggerAutomationSpec.Type.TYPE_SCHEDULE,
            schedule=automation,
        ),
    )

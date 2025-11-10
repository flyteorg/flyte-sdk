import asyncio
from typing import Union

from flyteidl2.core import interface_pb2, literals_pb2
from flyteidl2.task import common_pb2, run_pb2, task_definition_pb2
from google.protobuf import timestamp_pb2, wrappers_pb2

import flyte.types
from flyte import Cron, FixedRate, Trigger, TriggerTime


def _to_schedule(m: Union[Cron, FixedRate], kickoff_arg_name: str | None = None) -> common_pb2.Schedule:
    if isinstance(m, Cron):
        return common_pb2.Schedule(
            cron=common_pb2.Cron(
                expression=m.expression,
                timezone=m.timezone,
            ),
            kickoff_time_input_arg=kickoff_arg_name,
        )
    elif isinstance(m, FixedRate):
        start_time = None
        if m.start_time is not None:
            start_time = timestamp_pb2.Timestamp()
            start_time.FromDatetime(m.start_time)

        return common_pb2.Schedule(
            rate=common_pb2.FixedRate(
                value=m.interval_minutes,
                unit=common_pb2.FixedRateUnit.FIXED_RATE_UNIT_MINUTE,
                start_time=start_time,
            ),
            kickoff_time_input_arg=kickoff_arg_name,
        )


async def process_default_inputs(
    default_inputs: dict,
    task_name: str,
    task_inputs: interface_pb2.VariableMap,
    task_default_inputs: list[common_pb2.NamedParameter],
) -> list[common_pb2.NamedLiteral]:
    """
    Process default inputs and convert them to NamedLiteral objects.

    Args:
        default_inputs: Dictionary of default input values
        task_name: Name of the task for error messages
        task_inputs: Task input variable map
        task_default_inputs: List of default parameters from task

    Returns:
        List of NamedLiteral objects
    """
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

    literals: list[common_pb2.NamedLiteral] = []
    for k, lit in zip(keys, final_literals):
        literals.append(
            common_pb2.NamedLiteral(
                name=k,
                value=lit,
            )
        )

    return literals


async def to_task_trigger(
    t: Trigger,
    task_name: str,
    task_inputs: interface_pb2.VariableMap,
    task_default_inputs: list[common_pb2.NamedParameter],
) -> task_definition_pb2.TaskTrigger:
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
        env = run_pb2.Envs()
        for k, v in t.env_vars.items():
            env.values.append(literals_pb2.KeyValuePair(key=k, value=v))

    labels = run_pb2.Labels(values=t.labels) if t.labels else None

    annotations = run_pb2.Annotations(values=t.annotations) if t.annotations else None

    run_spec = run_pb2.RunSpec(
        overwrite_cache=t.overwrite_cache,
        envs=env,
        interruptible=wrappers_pb2.BoolValue(value=t.interruptible) if t.interruptible is not None else None,
        cluster=t.queue,
        labels=labels,
        annotations=annotations,
    )

    kickoff_arg_name = None
    default_inputs = {}
    if t.inputs:
        for k, v in t.inputs.items():
            if v is TriggerTime:
                if k == "trigger_time" and k not in task_inputs.variables:
                    # the 'trigger_time' input name that by default Triggers look for so it's always added.
                    # Remove it here by skipping if it's not actually an input to the task
                    continue
                kickoff_arg_name = k
            else:
                default_inputs[k] = v

    # assert that default_inputs and the kickoff_arg_name are infact in the task inputs
    if kickoff_arg_name is not None and kickoff_arg_name not in task_inputs.variables:
        raise ValueError(
            f"For a scheduled trigger, the TriggerTime input '{kickoff_arg_name}' "
            f"must be an input to the task, but not found in task {task_name}. "
            f"Available inputs: {list(task_inputs.variables.keys())}"
        )

    literals = await process_default_inputs(default_inputs, task_name, task_inputs, task_default_inputs)

    automation = _to_schedule(
        t.automation,
        kickoff_arg_name=kickoff_arg_name,
    )

    return task_definition_pb2.TaskTrigger(
        name=t.name,
        spec=task_definition_pb2.TaskTriggerSpec(
            active=t.auto_activate,
            run_spec=run_spec,
            inputs=common_pb2.Inputs(literals=literals),
        ),
        automation_spec=common_pb2.TriggerAutomationSpec(
            type=common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE,
            schedule=automation,
        ),
    )

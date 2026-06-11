import asyncio
from typing import Optional, Union

from flyteidl2.common import identifier_pb2
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.core import interface_pb2, literals_pb2
from flyteidl2.task import common_pb2, run_pb2, task_definition_pb2
from google.protobuf import timestamp_pb2, wrappers_pb2

import flyte.types
from flyte import Cron, FixedRate, Trigger, TriggerTime

# Reserved key under which the trigger's kickoff-time input arg name is stashed in the offloaded
# Inputs.context. Triggers now offload their inputs at registration (UploadInputs), so the backend
# no longer injects a kickoff-time literal at fire time and instead stamps run_start_time on the run.
# The runtime reads this key out of the offloaded inputs at execution and writes run_start_time into
# the named input, preserving the `inputs={"start_time": flyte.TriggerTime}` API. Context is not part
# of the cache-key hash, so carrying it here does not perturb input hashing.
KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY = "_u_kickoff_time_input_arg"


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
    # Convert variables list to dict for easier lookup
    variables_dict = {entry.key: entry.value for entry in task_inputs.variables}

    keys = []
    literal_coros = []
    for k, v in default_inputs.items():
        if k not in variables_dict:
            raise ValueError(
                f"Trigger default input '{k}' must be an input to the task, but not found in task {task_name}. "
                f"Available inputs: {list(variables_dict.keys())}"
            )
        else:
            literal_coros.append(flyte.types.TypeEngine.to_literal(v, type(v), variables_dict[k].type))
            keys.append(k)

    final_literals: list[literals_pb2.Literal] = await asyncio.gather(*literal_coros, return_exceptions=True)

    # Check for exceptions in the gathered results
    for k, lit in zip(keys, final_literals):
        if isinstance(lit, Exception):
            raise RuntimeError(f"Failed to convert trigger default input '{k}'") from lit

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

    notification_rule_name = None
    notification_rules = None
    if t.notifications:
        from .notifications_serde import resolve_notification_settings

        notification_rule_name, notification_rules = resolve_notification_settings(t.notifications)

    run_spec = run_pb2.RunSpec(
        overwrite_cache=t.overwrite_cache,
        envs=env,
        interruptible=wrappers_pb2.BoolValue(value=t.interruptible) if t.interruptible is not None else None,
        cluster=t.queue,
        max_action_concurrency=t.max_action_concurrency or 0,
        labels=labels,
        annotations=annotations,
        notification_rule_name=notification_rule_name,
        notification_rules=notification_rules,
    )

    kickoff_arg_name = None
    default_inputs = {}
    if t.inputs:
        for k, v in t.inputs.items():
            if v is TriggerTime:
                kickoff_arg_name = k
            else:
                default_inputs[k] = v

    # assert that default_inputs and the kickoff_arg_name are in fact in the task inputs
    # Convert variables list to dict for checking
    variables_dict = {entry.key: entry.value for entry in task_inputs.variables}
    if kickoff_arg_name is not None and kickoff_arg_name not in variables_dict:
        raise ValueError(
            f"For a scheduled trigger, the TriggerTime input '{kickoff_arg_name}' "
            f"must be an input to the task, but not found in task {task_name}. "
            f"Available inputs: {list(variables_dict.keys())}"
        )

    literals = await process_default_inputs(default_inputs, task_name, task_inputs, task_default_inputs)

    context_kvs: list[literals_pb2.KeyValuePair] = []
    if t.custom_context:
        context_kvs.extend(literals_pb2.KeyValuePair(key=k, value=v) for k, v in t.custom_context.items())

    # Also convey the kickoff-time input arg name through the (offloaded) inputs context, so the
    # runtime can fill that input from run_start_time at execution (the offloaded blob never carries
    # the per-fire value). Context is not part of the cache-key hash, so this does not perturb hashing.
    if kickoff_arg_name is not None:
        context_kvs.append(literals_pb2.KeyValuePair(key=KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, value=kickoff_arg_name))

    # Keep the kickoff arg on the schedule too: the backend uses it for the scheduled-trigger
    # contract (and folds run_start_time into the cache key on fire).
    automation = _to_schedule(t.automation, kickoff_arg_name=kickoff_arg_name)

    return task_definition_pb2.TaskTrigger(
        name=t.name,
        spec=task_definition_pb2.TaskTriggerSpec(
            active=t.auto_activate,
            run_spec=run_spec,
            inputs=common_pb2.Inputs(literals=literals, context=context_kvs),
            description=t.description,
        ),
        automation_spec=common_pb2.TriggerAutomationSpec(
            type=common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE,
            schedule=automation,
        ),
    )


async def offload_trigger_inputs(
    inputs: common_pb2.Inputs,
    *,
    org: Optional[str],
    project: Optional[str],
    domain: Optional[str],
    task_version: str,
    task_name: Optional[str] = None,
    task_spec: Optional[task_definition_pb2.TaskSpec] = None,
) -> Optional[common_run_pb2.OffloadedInputData]:
    """Offload trigger inputs out-of-band via DataProxy and return the URI + hash, or None.

    Routing goes through SelectCluster's ``OPERATION_UPLOAD_TRIGGER`` (zero-trust path). When the
    backend does not have zero trust enabled it returns ``UNIMPLEMENTED`` for that operation; we
    catch it and return ``None`` so the caller falls back to inline trigger inputs (the pre-offload
    flow).

    The ``task`` reference is only used by the server to resolve the task template's
    ``cache_ignore_input_vars`` so the input hash matches a later launch; it stores nothing
    trigger-specific. ``project_id`` supplies the storage location (org/project/domain prefix);
    no trigger id is involved, since offloaded inputs are content-addressed by hash and referenced
    by URI from the trigger spec.

    Pass ``task_spec`` when the task is not yet registered (deploy path: the task is being created in
    the same request, so a ``task_id`` lookup would 404). Pass ``task_name`` to reference an
    already-registered task by id (``remote.Trigger.create`` path).
    """
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from flyteidl2.dataproxy import dataproxy_service_pb2

    from flyte._initialize import get_client

    req = dataproxy_service_pb2.UploadInputsRequest(
        inputs=inputs,
        project_id=identifier_pb2.ProjectIdentifier(organization=org, name=project, domain=domain),
    )
    if task_spec is not None:
        req.task_spec.CopyFrom(task_spec)
    elif task_name is not None:
        req.task_id.CopyFrom(
            task_definition_pb2.TaskIdentifier(
                org=org, project=project, domain=domain, name=task_name, version=task_version
            )
        )
    else:
        raise ValueError("offload_trigger_inputs requires either task_spec or task_name")

    try:
        resp = await get_client().dataproxy_service.upload_trigger(req)
    except ConnectError as e:
        if e.code == Code.UNIMPLEMENTED:
            # Zero trust is not enabled on the backend; fall back to inline trigger inputs.
            return None
        raise
    return resp.offloaded_input_data

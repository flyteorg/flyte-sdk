import hashlib
import os
import posixpath
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from flyte import storage
from flyte import system_logger as logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import GetTaskLogsResponse, GetTaskLogsResponseBody, TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp

from flyteplugins.slurm.script import render_container_job, render_script_job
from flyteplugins.slurm.transport import SlurmJobState, SlurmTransport, SSHTransport

TASK_TYPE_NATIVE = "slurm"
TASK_TYPE_SCRIPT = "slurm_script"

# Connector-level defaults so a platform team can configure the cluster once on the
# flyteconnector deployment instead of on every task.
ENV_HOST = "FLYTE_SLURM_HOST"
ENV_PORT = "FLYTE_SLURM_PORT"
ENV_USERNAME = "FLYTE_SLURM_USERNAME"
ENV_SSH_PRIVATE_KEY = "FLYTE_SLURM_SSH_PRIVATE_KEY"
ENV_KNOWN_HOSTS = "FLYTE_SLURM_KNOWN_HOSTS"
ENV_WORKING_DIR = "FLYTE_SLURM_WORKING_DIR"
ENV_SKIP_HOST_KEY_VERIFICATION = "FLYTE_SLURM_SKIP_HOST_KEY_VERIFICATION"

DEFAULT_WORKING_DIR = ".flyte/jobs"
_STDERR_TAIL_LINES = 30
_LOG_TAIL_LINES = 500

# Slurm job states -> Flyte phases. Only the first token of the state is matched,
# so "CANCELLED by 1234" maps the same as "CANCELLED".
_QUEUED = {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SUSPENDED", "STOPPED"}
_RUNNING = {"RUNNING", "COMPLETING", "STAGE_OUT", "SIGNALING", "RESIZING"}
_SUCCEEDED = {"COMPLETED"}
_FAILED = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "TIMEOUT", "DEADLINE", "BOOT_FAIL", "SPECIAL_EXIT", "REVOKED"}
_RETRYABLE = {"PREEMPTED"}
_ABORTED = {"CANCELLED"}


def slurm_state_to_phase(state: str) -> TaskExecution.Phase:
    base = (state or "").split()[0].upper() if state else ""
    if base in _QUEUED:
        return TaskExecution.QUEUED
    if base in _RUNNING:
        return TaskExecution.RUNNING
    if base in _SUCCEEDED:
        return TaskExecution.SUCCEEDED
    if base in _RETRYABLE:
        return TaskExecution.RETRYABLE_FAILED
    if base in _ABORTED:
        return TaskExecution.ABORTED
    if base in _FAILED:
        return TaskExecution.FAILED
    # Polling on would run until the task's own timeout with nothing explaining why, and
    # an unknown state is more often terminal than not.
    raise ValueError(
        f"Unrecognized Slurm job state {state!r}. If this is a real Slurm state, it needs adding to "
        "the state map in flyteplugins.slurm.connector."
    )


@dataclass
class SlurmJobMetadata(ResourceMeta):
    job_id: str
    job_name: str
    host: str
    username: str
    port: int
    stdout_path: str
    stderr_path: str
    known_hosts: Optional[str] = None
    skip_host_key_verification: bool = False
    #: Declared output name -> (uri, kind). Empty for native tasks and for script tasks
    #: that declare none. Recorded at create time so `get` does not re-derive the prefix.
    declared_outputs: Dict[str, List[str]] = field(default_factory=dict)


def _job_name(task_template: TaskTemplate, tem: Optional[TaskExecutionMetadata]) -> str:
    base = task_template.id.name.rsplit(".", 1)[-1] if task_template.id.name else "task"
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", base)[:40].strip("-.") or "task"
    return f"flyte-{base}-{uuid.uuid4().hex[:8]}"


def _int_or_none(value: Any) -> Any:
    # Values round-trip through a protobuf Struct, which turns every number into a float.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _env_from_template(task_template: TaskTemplate) -> Dict[str, str]:
    if task_template.container is None:
        return {}
    return {kv.key: kv.value for kv in task_template.container.env}


def _env_from_execution_id(tem: Optional[TaskExecutionMetadata]) -> Dict[str, str]:
    """Execution-identity variables that `flytek8s` adds when it builds a pod.

    `GetExecutionEnvVars` (`flyte2/flyteplugins/go/tasks/pluginmachinery/flytek8s/
    k8s_resource_adds.go`) writes these into every task pod, but that code runs only on
    the Kubernetes path -- a connector never benefits from it. The runtime asserts on
    `org`, `project`, `domain`, `run_name` and `name`; the connector metadata's env
    supplies org/run/action, leaving project and domain to be derived here from the
    execution identifier, or the entrypoint aborts with `Project is required`.
    """
    if tem is None or not tem.HasField("task_execution_id"):
        return {}
    execution = tem.task_execution_id.node_execution_id.execution_id
    env: Dict[str, str] = {}
    if execution.project:
        env["FLYTE_INTERNAL_EXECUTION_PROJECT"] = execution.project
    if execution.domain:
        env["FLYTE_INTERNAL_EXECUTION_DOMAIN"] = execution.domain
    if execution.name:
        env["FLYTE_INTERNAL_EXECUTION_ID"] = execution.name
    if execution.org:
        env["_U_ORG_NAME"] = execution.org
    env["FLYTE_ATTEMPT_NUMBER"] = str(tem.task_execution_id.retry_attempt)
    return env


def _env_from_platform(tem: Optional[TaskExecutionMetadata]) -> Dict[str, str]:
    """Platform variables the backend injects into a task container.

    On Kubernetes the leaseworker writes these straight into the pod spec
    (`leaseworker/lifecycle/context.go`): `_U_RUN_BASE` (the run's output prefix),
    `ACTION_NAME`, `RUN_NAME`, `_U_ORG_NAME` and the endpoint-discovery pair
    `_U_EP_OVERRIDE` / `_U_INSECURE`. They are *not* part of the TaskTemplate, so a
    connector only sees them through `TaskExecutionMetadata.environment_variables`.

    Dropping them is not cosmetic: `a0` requires `--run-base-dir`, whose only other
    source is `_U_RUN_BASE`, so without this the entrypoint exits 2 with
    `Missing option '--run-base-dir'`.
    """
    if tem is None:
        return {}
    return dict(tem.environment_variables)


def _output_destinations(outputs: Dict[str, str], output_prefix: str) -> Dict[str, List[str]]:
    """Destination URI per declared output, under the action's output prefix.

    The same prefix `outputs.pb` would go to for a native task, so a script task's
    results land where the rest of the action's data lives.
    """
    return {name: [f"{output_prefix.rstrip('/')}/{name}", kind] for name, kind in outputs.items()}


def _env_from_outputs(destinations: Dict[str, List[str]]) -> Dict[str, str]:
    """`FLYTE_OUTPUT_<NAME>` for each declared output.

    A script cannot write Flyte's output format, so it is told where to put each result
    and writes there with its own tooling. Mirrors how inputs arrive.
    """
    return {
        "FLYTE_OUTPUT_" + re.sub(r"[^A-Za-z0-9_]", "_", name).upper(): uri for name, (uri, _) in destinations.items()
    }


def _env_from_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Expose scalar task inputs to a script job as FLYTE_INPUT_<NAME>."""
    env: Dict[str, str] = {}
    for key, value in (inputs or {}).items():
        if isinstance(value, (str, int, float, bool)):
            name = "FLYTE_INPUT_" + re.sub(r"[^A-Za-z0-9_]", "_", key).upper()
            env[name] = str(value).lower() if isinstance(value, bool) else str(value)
        elif hasattr(value, "path"):
            # File and Dir carry a URI the script can fetch with its own tooling.
            env["FLYTE_INPUT_" + re.sub(r"[^A-Za-z0-9_]", "_", key).upper()] = str(value.path)
        else:
            # Warning into the connector's log would not reach whoever wrote the task, and
            # a silently unset variable is worse than a failed submission.
            raise ValueError(
                f"Input {key!r} of type {type(value).__name__} cannot be passed to an sbatch script. "
                "Scalars become FLYTE_INPUT_<NAME>, and File/Dir become their URI; anything else has "
                "no representation in the environment. Pass a URI as a string and fetch it in the script."
            )
    return env


class SlurmConnector(AsyncConnector[SlurmJobMetadata]):
    """Run Flyte tasks as Slurm jobs.

    `slurm` submits the task's own container image and Flyte entrypoint via Pyxis/Enroot,
    so typed I/O, caching and retries work exactly as they do for a Kubernetes pod.
    `slurm_script` submits a user-supplied sbatch script as-is and reports phase only.
    """

    name: str = "Slurm Connector"
    task_type_name: str = TASK_TYPE_NATIVE
    metadata_type: type = SlurmJobMetadata

    def __init__(self):
        self._transports: Dict[str, SlurmTransport] = {}

    # ---- transport plumbing ----

    def _transport(
        self,
        host: str,
        port: int,
        username: str,
        private_key: str,
        known_hosts: Optional[str],
        skip_host_key_verification: bool,
    ) -> SlurmTransport:
        key_digest = hashlib.sha256(private_key.encode()).hexdigest()[:16]
        cache_key = f"{username}@{host}:{port}/{key_digest}/{known_hosts}/{skip_host_key_verification}"
        transport = self._transports.get(cache_key)
        if transport is None:
            transport = SSHTransport(
                host=host,
                port=port,
                username=username,
                private_key=private_key,
                known_hosts=known_hosts,
                skip_host_key_verification=skip_host_key_verification,
            )
            self._transports[cache_key] = transport
        return transport

    def _transport_for_meta(self, meta: SlurmJobMetadata, ssh_private_key: Optional[str]) -> SlurmTransport:
        key = ssh_private_key or os.getenv(ENV_SSH_PRIVATE_KEY)
        if not key:
            raise ValueError(
                "Missing Slurm SSH private key. Set `ssh_private_key` on the Slurm config to the name of a "
                f"Flyte secret, or set {ENV_SSH_PRIVATE_KEY} on the connector."
            )
        return self._transport(
            meta.host, meta.port, meta.username, key, meta.known_hosts, meta.skip_host_key_verification
        )

    # ---- connector interface ----

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        ssh_private_key: Optional[str] = None,
        **kwargs,
    ) -> SlurmJobMetadata:
        custom = MessageToDict(task_template.custom)
        connection = custom.get("connection") or {}

        # The connector's environment wins over task config. The SSH key belongs to the
        # deployment and is shared by every task, so letting a task redirect it to a host
        # of its choosing would hand that key to whoever wrote the task -- more so with
        # host-key verification disabled. Task config still supplies these on a connector
        # that sets none of them, which is how local execution works.
        host = os.getenv(ENV_HOST) or connection.get("host")
        username = os.getenv(ENV_USERNAME) or connection.get("username")
        port = int(os.getenv(ENV_PORT) or _int_or_none(connection.get("port")) or 22)
        known_hosts = os.getenv(ENV_KNOWN_HOSTS) or connection.get("known_hosts")
        # Never task-settable: a task could otherwise turn off host-key checking for a
        # connection made with the deployment's key.
        skip_host_key_verification = os.getenv(ENV_SKIP_HOST_KEY_VERIFICATION, "").lower() in ("1", "true", "yes")
        if connection.get("skip_host_key_verification") and not skip_host_key_verification:
            logger.warning(
                "Ignoring `skip_host_key_verification` from task config; set "
                f"{ENV_SKIP_HOST_KEY_VERIFICATION} on the connector if that is really wanted."
            )
        if not host or not username:
            raise ValueError(
                "Missing Slurm connection details. Set `host` and `username` on the Slurm config, "
                f"or set {ENV_HOST} and {ENV_USERNAME} on the connector."
            )

        meta = SlurmJobMetadata(
            job_id="",
            job_name=_job_name(task_template, task_execution_metadata),
            host=host,
            username=username,
            port=port,
            stdout_path="",
            stderr_path="",
            known_hosts=known_hosts,
            skip_host_key_verification=skip_host_key_verification,
        )
        transport = self._transport_for_meta(meta, ssh_private_key)

        working_dir = custom.get("working_dir") or os.getenv(ENV_WORKING_DIR) or DEFAULT_WORKING_DIR
        if not posixpath.isabs(working_dir):
            working_dir = posixpath.join(await transport.home(), working_dir)  # type: ignore[attr-defined]
        script_path = posixpath.join(working_dir, f"{meta.job_name}.sbatch")
        meta.stdout_path = posixpath.join(working_dir, f"{meta.job_name}.out")
        meta.stderr_path = posixpath.join(working_dir, f"{meta.job_name}.err")

        sbatch_fields = {k: _int_or_none(v) for k, v in (custom.get("sbatch") or {}).items()}
        sbatch_extra = {k: _int_or_none(v) for k, v in (custom.get("sbatch_options") or {}).items()}
        # Precedence, lowest first: template env, identity derived from the execution id,
        # the platform vars the backend sent explicitly, then the task's own config.
        env = {
            **_env_from_template(task_template),
            **_env_from_execution_id(task_execution_metadata),
            **_env_from_platform(task_execution_metadata),
            **(custom.get("env") or {}),
        }

        if task_template.type == TASK_TYPE_SCRIPT:
            script_body = custom.get("script")
            if not script_body:
                raise ValueError("slurm_script task has no script")
            env.update(_env_from_inputs(inputs))
            meta.declared_outputs = _output_destinations(custom.get("outputs") or {}, output_prefix)
            env.update(_env_from_outputs(meta.declared_outputs))
            script = render_script_job(
                job_name=meta.job_name,
                stdout_path=meta.stdout_path,
                stderr_path=meta.stderr_path,
                script=script_body,
                env=env,
                sbatch_fields=sbatch_fields,
                sbatch_extra=sbatch_extra,
                modules=custom.get("modules") or [],
            )
        else:
            container_cfg = custom.get("container") or {}
            container = task_template.container
            if container is None:
                raise ValueError("slurm task has no container spec")
            image = container_cfg.get("image") or container.image
            if not image:
                raise ValueError("slurm task has no container image")
            script = render_container_job(
                job_name=meta.job_name,
                stdout_path=meta.stdout_path,
                stderr_path=meta.stderr_path,
                image=image,
                command=[*container.command, *container.args],
                env=env,
                sbatch_fields=sbatch_fields,
                sbatch_extra=sbatch_extra,
                container_mounts=container_cfg.get("mounts") or [],
                container_workdir=container_cfg.get("workdir"),
                srun_extra_args=container_cfg.get("srun_args") or [],
                container_runtime=container_cfg.get("runtime") or "pyxis",
                container_args=container_cfg.get("args") or [],
                modules=custom.get("modules") or [],
            )

        # Deliberately not the script body: it carries every exported variable, including
        # _U_RUN_BASE and whatever the deployment injects. The script is on the cluster at
        # `script_path` for anyone who needs to read it.
        logger.debug(f"Submitting Slurm job {meta.job_name} to {username}@{host} from {script_path}")
        meta.job_id = await transport.submit(script, script_path)
        logger.info(f"Submitted Slurm job {meta.job_id} ({meta.job_name}) to {host}")
        return meta

    async def get(self, resource_meta: SlurmJobMetadata, ssh_private_key: Optional[str] = None, **kwargs) -> Resource:
        transport = self._transport_for_meta(resource_meta, ssh_private_key)
        states = await transport.status([resource_meta.job_id])
        state = states.get(resource_meta.job_id)
        if state is None:
            raise RuntimeError(
                f"Slurm job {resource_meta.job_id} ({resource_meta.job_name}) is not known to squeue or sacct "
                f"on {resource_meta.host}"
            )

        phase = slurm_state_to_phase(state.base_state)

        # The job's stdout and stderr are files on the login node, not resources behind a
        # URL. A TaskLog uri renders as a hyperlink, so returning a POSIX path there makes a
        # dead link in the UI; name the paths in the message instead. Streaming stdout is
        # `get_logs`' job. Kept ahead of the stderr tail so that block stays last.
        message = (
            f"{_describe(state)}\n"
            f"Job files on {resource_meta.username}@{resource_meta.host}: "
            f"{resource_meta.stdout_path} (stdout), {resource_meta.stderr_path} (stderr)"
        )
        outputs: Optional[Dict[str, Any]] = None
        if phase == TaskExecution.SUCCEEDED and resource_meta.declared_outputs:
            outputs = await _collect_outputs(resource_meta)

        if phase == TaskExecution.RUNNING:
            # Slurm gives interactive access to a running allocation for free, so point at
            # it. Flyte's own debug SSH cannot help here: the entrypoint would start a
            # server inside the job, but nothing routes to a Slurm node.
            message += (
                f"\nAttach to the running job: ssh {resource_meta.username}@{resource_meta.host} "
                f"'srun --jobid={resource_meta.job_id} --overlap --pty bash'"
            )
        if phase in (TaskExecution.FAILED, TaskExecution.RETRYABLE_FAILED):
            tail = await transport.tail(resource_meta.stderr_path, _STDERR_TAIL_LINES)
            if tail.strip():
                message = f"{message}\n--- stderr (last {_STDERR_TAIL_LINES} lines) ---\n{tail.rstrip()}"

        return Resource(phase=phase, message=message, outputs=outputs)

    async def delete(self, resource_meta: SlurmJobMetadata, ssh_private_key: Optional[str] = None, **kwargs):
        transport = self._transport_for_meta(resource_meta, ssh_private_key)
        await transport.cancel(resource_meta.job_id)

    async def get_logs(
        self, resource_meta: SlurmJobMetadata, ssh_private_key: Optional[str] = None, **kwargs
    ) -> AsyncIterator[GetTaskLogsResponse]:
        from flyteidl2.logs.dataplane.payload_pb2 import LogLine

        transport = self._transport_for_meta(resource_meta, ssh_private_key)
        text = await transport.tail(resource_meta.stdout_path, _LOG_TAIL_LINES)
        now = Timestamp()
        now.FromDatetime(datetime.now(timezone.utc))
        lines = [LogLine(timestamp=now, message=line) for line in text.splitlines()]
        yield GetTaskLogsResponse(body=GetTaskLogsResponseBody(lines=lines))


class SlurmScriptConnector(SlurmConnector):
    task_type_name: str = TASK_TYPE_SCRIPT


async def _collect_outputs(resource_meta: SlurmJobMetadata) -> Dict[str, Any]:
    """Turn each declared destination into a File or Dir, failing if nothing was written.

    A script that exits 0 without writing a declared output would otherwise hand a
    downstream task a URI to nothing, which surfaces much later as a confusing read
    error. Checking here costs one existence call per output, once, on the poll that
    finds the job finished.
    """
    from flyte.io import Dir, File

    collected: Dict[str, Any] = {}
    missing = []
    for name, (uri, kind) in resource_meta.declared_outputs.items():
        if not await storage.exists(uri):
            missing.append((name, uri))
            continue
        collected[name] = Dir.from_existing_remote(uri) if kind == "directory" else File.from_existing_remote(uri)

    if missing:
        listed = "; ".join(f"{name} at {uri}" for name, uri in missing)
        raise RuntimeError(
            f"Slurm job {resource_meta.job_id} succeeded but did not write every declared output: {listed}. "
            "The script is given each destination as FLYTE_OUTPUT_<NAME> and must write there."
        )
    return collected


def _describe(state: SlurmJobState) -> str:
    parts = [f"Slurm job {state.job_id} is {state.state}"]
    if state.exit_code and state.exit_code != "0:0":
        parts.append(f"exit code {state.exit_code}")
    if state.reason:
        parts.append(f"reason: {state.reason}")
    return ", ".join(parts)


ConnectorRegistry.register(SlurmConnector())
ConnectorRegistry.register(SlurmScriptConnector())

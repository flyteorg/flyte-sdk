import hashlib
import os
import posixpath
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

from flyte import system_logger as logger
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import GetTaskLogsResponse, GetTaskLogsResponseBody, TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
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
    logger.warning(f"Unrecognized Slurm job state {state!r}; treating as RUNNING")
    return TaskExecution.RUNNING


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


def _env_from_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Expose scalar task inputs to a script job as FLYTE_INPUT_<NAME>."""
    env: Dict[str, str] = {}
    for key, value in (inputs or {}).items():
        if isinstance(value, (str, int, float, bool)):
            name = "FLYTE_INPUT_" + re.sub(r"[^A-Za-z0-9_]", "_", key).upper()
            env[name] = str(value).lower() if isinstance(value, bool) else str(value)
    return env


class SlurmConnector(AsyncConnector[SlurmJobMetadata]):
    """Run Flyte tasks as Slurm jobs.

    ``slurm`` submits the task's own container image and Flyte entrypoint via Pyxis/Enroot,
    so typed I/O, caching and retries work exactly as they do for a Kubernetes pod.
    ``slurm_script`` submits a user-supplied sbatch script as-is and reports phase only.
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

        host = connection.get("host") or os.getenv(ENV_HOST)
        username = connection.get("username") or os.getenv(ENV_USERNAME)
        port = int(_int_or_none(connection.get("port")) or os.getenv(ENV_PORT) or 22)  # task config wins
        known_hosts = connection.get("known_hosts") or os.getenv(ENV_KNOWN_HOSTS)
        skip_host_key_verification = bool(connection.get("skip_host_key_verification", False))
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
        env = {**_env_from_template(task_template), **(custom.get("env") or {})}

        if task_template.type == TASK_TYPE_SCRIPT:
            script_body = custom.get("script")
            if not script_body:
                raise ValueError("slurm_script task has no script")
            env.update(_env_from_inputs(inputs))
            script = render_script_job(
                job_name=meta.job_name,
                stdout_path=meta.stdout_path,
                stderr_path=meta.stderr_path,
                script=script_body,
                env=env,
                sbatch_fields=sbatch_fields,
                sbatch_extra=sbatch_extra,
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
            )

        logger.debug(f"Submitting Slurm job {meta.job_name} to {username}@{host}:\n{script}")
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
        message = _describe(state)
        if phase in (TaskExecution.FAILED, TaskExecution.RETRYABLE_FAILED):
            tail = await transport.tail(resource_meta.stderr_path, _STDERR_TAIL_LINES)
            if tail.strip():
                message = f"{message}\n--- stderr (last {_STDERR_TAIL_LINES} lines) ---\n{tail.rstrip()}"

        log_links = [
            TaskLog(uri=resource_meta.stdout_path, name="Slurm stdout"),
            TaskLog(uri=resource_meta.stderr_path, name="Slurm stderr"),
        ]
        return Resource(phase=phase, message=message, log_links=log_links)

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


def _describe(state: SlurmJobState) -> str:
    parts = [f"Slurm job {state.job_id} is {state.state}"]
    if state.exit_code and state.exit_code != "0:0":
        parts.append(f"exit code {state.exit_code}")
    if state.reason:
        parts.append(f"reason: {state.reason}")
    return ", ".join(parts)


ConnectorRegistry.register(SlurmConnector())
ConnectorRegistry.register(SlurmScriptConnector())

"""Transports the connector uses to talk to a Slurm cluster.

Phase 1 ships SSH to a login node, which works for any Slurm cluster and is
the path Soperator exposes by default. `SlurmTransport` is the seam for a
slurmrestd transport later; the connector only depends on the protocol.
"""

from __future__ import annotations

import asyncio
import posixpath
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Protocol, Tuple

from flyte import system_logger as logger

# sacct/squeue emit "CANCELLED by 1234"; only the first token is the state.
_SACCT_FORMAT = "JobIDRaw,State,ExitCode,Reason"
_SQUEUE_FORMAT = "%i|%T|%r"


@dataclass
class SlurmJobState:
    job_id: str
    state: str
    exit_code: Optional[str] = None
    reason: Optional[str] = None

    @property
    def base_state(self) -> str:
        """The bare state name, with everything Slurm decorates it with removed.

        `sacct` and `squeue` both embellish: `CANCELLED by 1234` carries the cancelling
        uid, and a state is suffixed with `+` when the field was truncated or carries
        extra information (`CANCELLED+`). Neither form matches a bare state name, so
        without this a terminal job falls through to the unrecognized-state path and is
        reported as RUNNING -- leaving the task polling until its timeout.
        """
        if not self.state:
            return ""
        return self.state.split()[0].rstrip("+").upper()


class SlurmTransport(Protocol):
    async def submit(self, script: str, script_path: str) -> str:
        """Upload `script` to `script_path` on the cluster, submit it, and return the job id."""
        ...

    async def status(self, job_ids: Iterable[str]) -> Dict[str, SlurmJobState]:
        """Return the state of each job that Slurm still knows about."""
        ...

    async def cancel(self, job_id: str) -> None:
        """Cancel a job. Must be idempotent."""
        ...

    async def tail(self, path: str, lines: int) -> str:
        """Return the last `lines` lines of a file on the cluster, or "" if it does not exist."""
        ...


def parse_sacct(output: str) -> Dict[str, SlurmJobState]:
    """Parse `sacct --parsable2 --noheader --format=JobIDRaw,State,ExitCode,Reason` output."""
    states: Dict[str, SlurmJobState] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        # With -X we only get allocations, but be defensive about step rows ("123.batch").
        if "." in job_id:
            continue
        states[job_id] = SlurmJobState(
            job_id=job_id,
            state=parts[1].strip(),
            exit_code=parts[2].strip() if len(parts) > 2 and parts[2].strip() else None,
            reason=parts[3].strip() if len(parts) > 3 and parts[3].strip() not in ("", "None") else None,
        )
    return states


def parse_squeue(output: str) -> Dict[str, SlurmJobState]:
    """Parse `squeue -h -o '%i|%T|%r'` output."""
    states: Dict[str, SlurmJobState] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id = parts[0].strip()
        reason = parts[2].strip() if len(parts) > 2 else ""
        states[job_id] = SlurmJobState(
            job_id=job_id,
            state=parts[1].strip(),
            reason=reason if reason and reason != "None" else None,
        )
    return states


_UNAVAILABLE_MARKERS = (
    "unable to contact",
    "connection timed out",
    "socket timed out",
    "slurmdbd",
    "no such file or directory",
    "protocol authentication error",
)


def _raise_if_unavailable(command: str, host: str, stderr: str, rc: int) -> None:
    """Fail loudly when Slurm itself is unreachable.

    A non-zero exit means either "no such job", which is ordinary, or "the controller is
    down", which is not. They are told apart by stderr: treating the second as an absent
    job reports a running job as gone and fails the task.
    """
    if rc == 0:
        return
    lowered = stderr.lower()
    if any(marker in lowered for marker in _UNAVAILABLE_MARKERS):
        raise RuntimeError(f"`{command}` could not reach Slurm on {host} (rc={rc}): {stderr.strip()}")


def parse_scontrol(output: str) -> Optional[SlurmJobState]:
    """Parse `scontrol show job <id> --oneliner` into a state.

    Used only when a job is in neither squeue nor sacct, which happens on clusters
    without accounting. scontrol keeps a finished job for MinJobAge seconds.
    """
    fields = dict(part.split("=", 1) for part in output.split() if "=" in part and not part.startswith("="))
    job_id = fields.get("JobId")
    state = fields.get("JobState")
    if not job_id or not state:
        return None
    reason = fields.get("Reason")
    return SlurmJobState(
        job_id=job_id,
        state=state,
        exit_code=fields.get("ExitCode"),
        reason=None if reason in (None, "None") else reason,
    )


def parse_sbatch_job_id(output: str) -> str:
    """`sbatch --parsable` prints `<jobid>` or `<jobid>;<cluster>`."""
    first = output.strip().splitlines()[0] if output.strip() else ""
    job_id = first.split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {output!r}")
    return job_id


class SSHTransport:
    """Drive Slurm through `sbatch`/`squeue`/`sacct`/`scancel` over SSH.

    One connection per (host, port, username) is kept open and reused across calls and
    re-established transparently if it drops, so tracking many jobs costs one login-node
    session rather than one per job.

    `status` accepts several job ids and queries them in a single `squeue`, but the
    connector currently passes one id at a time: `AsyncConnector.get` is called per
    resource, so coalescing would need a cache in the connector. The batching parameter
    is kept because that layer belongs here, not in the caller.
    """

    def __init__(
        self,
        host: str,
        username: str,
        private_key: str,
        port: int = 22,
        known_hosts: Optional[str] = None,
        known_hosts_data: Optional[str] = None,
        skip_host_key_verification: bool = False,
        connect_timeout: float = 30.0,
        command_timeout: float = 60.0,
    ):
        if not host:
            raise ValueError("Slurm SSH transport requires a host")
        if not username:
            raise ValueError("Slurm SSH transport requires a username")
        if not private_key:
            raise ValueError("Slurm SSH transport requires a private key")
        self._host = host
        self._port = port
        self._username = username
        self._private_key = private_key
        self._known_hosts = known_hosts
        self._known_hosts_data = known_hosts_data
        self._skip_host_key_verification = skip_host_key_verification
        self._connect_timeout = connect_timeout
        self._command_timeout = command_timeout
        self._conn = None
        self._lock = asyncio.Lock()

    async def _connection(self):
        import asyncssh

        async with self._lock:
            if self._conn is not None and not self._conn.is_closed():
                return self._conn
            if self._skip_host_key_verification:
                known_hosts = None
                logger.warning(
                    f"Slurm SSH transport: host key verification disabled for {self._host}. "
                    "Set `known_hosts` on the Slurm config for production use."
                )
            elif self._known_hosts_data:
                # asyncssh parses bytes as known_hosts content rather than a filename, so
                # the entries can come from a Flyte secret instead of a file the
                # deployment has to mount.
                known_hosts = self._known_hosts_data.encode()
            else:
                # () means asyncssh's default (~/.ssh/known_hosts on the connector).
                known_hosts = self._known_hosts or ()
            key = asyncssh.import_private_key(self._private_key)
            self._conn = await asyncio.wait_for(
                asyncssh.connect(
                    self._host,
                    port=self._port,
                    username=self._username,
                    client_keys=[key],
                    known_hosts=known_hosts,
                ),
                timeout=self._connect_timeout,
            )
            return self._conn

    async def _run(self, command: str, check: bool = True) -> Tuple[str, str, int]:
        conn = await self._connection()
        try:
            # A wedged login node would otherwise block the poll forever, and with it
            # every other job this connector is tracking.
            result = await asyncio.wait_for(conn.run(command, check=False), timeout=self._command_timeout)
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"`{command}` timed out after {self._command_timeout:.0f}s on {self._host}") from e
        stdout = result.stdout if isinstance(result.stdout, str) else (result.stdout or b"").decode()
        stderr = result.stderr if isinstance(result.stderr, str) else (result.stderr or b"").decode()
        rc = -1 if result.exit_status is None else result.exit_status
        if check and rc != 0:
            raise RuntimeError(f"`{command}` failed on {self._host} (rc={rc}): {stderr.strip() or stdout.strip()}")
        return stdout, stderr, rc

    async def submit(self, script: str, script_path: str) -> str:
        import asyncssh

        conn = await self._connection()
        await self._run(f"mkdir -p {_q(posixpath.dirname(script_path))}")
        async with conn.start_sftp_client() as sftp:
            async with sftp.open(script_path, "w") as f:
                await f.write(script)
            await sftp.chmod(script_path, 0o700)
        try:
            stdout, _, _ = await self._run(f"sbatch --parsable {_q(script_path)}")
        except asyncssh.Error as e:
            raise RuntimeError(f"sbatch failed on {self._host}: {e}") from e
        return parse_sbatch_job_id(stdout)

    async def status(self, job_ids: Iterable[str]) -> Dict[str, SlurmJobState]:
        ids = sorted({str(j) for j in job_ids})
        if not ids:
            return {}
        joined = ",".join(ids)
        # squeue is authoritative for jobs still in the system; sacct fills in the ones
        # that finished. Both exit non-zero when none of the ids exist, which is not an
        # error for us -- but so does an unreachable controller, and treating that as
        # "job gone" would report a running job as vanished.
        stdout, stderr, rc = await self._run(f"squeue -h -j {joined} -o {_q(_SQUEUE_FORMAT)}", check=False)
        _raise_if_unavailable("squeue", self._host, stderr, rc)
        states = parse_squeue(stdout)

        missing = [j for j in ids if j not in states]
        if missing:
            stdout, stderr, rc = await self._run(
                f"sacct -X -n -P -j {','.join(missing)} --format={_SACCT_FORMAT}", check=False
            )
            _raise_if_unavailable("sacct", self._host, stderr, rc)
            states.update(parse_sacct(stdout))

        # Last resort for a cluster without accounting, where a finished job leaves
        # squeue and is unknown to sacct. scontrol keeps a job in memory briefly after it
        # ends (MinJobAge), which is usually long enough to catch the transition.
        missing = [j for j in ids if j not in states]
        for job_id in missing:
            stdout, _, rc = await self._run(f"scontrol show job {_q(job_id)} --oneliner", check=False)
            if rc == 0 and stdout.strip():
                state = parse_scontrol(stdout)
                if state:
                    states[job_id] = state
        return states

    async def cancel(self, job_id: str) -> None:
        _, stderr, rc = await self._run(f"scancel {_q(str(job_id))}", check=False)
        # Already-finished or unknown jobs are fine: cancel must be idempotent.
        if rc != 0 and "Invalid job id" not in stderr:
            raise RuntimeError(f"scancel {job_id} failed on {self._host}: {stderr.strip()}")

    async def tail(self, path: str, lines: int = 100) -> str:
        stdout, _, rc = await self._run(f"tail -n {int(lines)} {_q(path)} 2>/dev/null", check=False)
        return stdout if rc == 0 else ""

    async def home(self) -> str:
        """Absolute home directory of the SSH user, used to anchor relative working directories."""
        stdout, _, _ = await self._run("pwd")
        return stdout.strip()

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _q(value: str) -> str:
    import shlex

    return shlex.quote(value)

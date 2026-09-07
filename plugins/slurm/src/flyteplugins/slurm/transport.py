"""Transports the connector uses to talk to a Slurm cluster.

Phase 1 ships SSH to a login node, which works for any Slurm cluster and is
the path Soperator exposes by default. ``SlurmTransport`` is the seam for a
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
        return self.state.split()[0].upper() if self.state else ""


class SlurmTransport(Protocol):
    async def submit(self, script: str, script_path: str) -> str:
        """Upload ``script`` to ``script_path`` on the cluster, submit it, and return the job id."""
        ...

    async def status(self, job_ids: Iterable[str]) -> Dict[str, SlurmJobState]:
        """Return the state of each job that Slurm still knows about."""
        ...

    async def cancel(self, job_id: str) -> None:
        """Cancel a job. Must be idempotent."""
        ...

    async def tail(self, path: str, lines: int) -> str:
        """Return the last ``lines`` lines of a file on the cluster, or "" if it does not exist."""
        ...


def parse_sacct(output: str) -> Dict[str, SlurmJobState]:
    """Parse ``sacct --parsable2 --noheader --format=JobIDRaw,State,ExitCode,Reason`` output."""
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
    """Parse ``squeue -h -o '%i|%T|%r'`` output."""
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


def parse_sbatch_job_id(output: str) -> str:
    """``sbatch --parsable`` prints ``<jobid>`` or ``<jobid>;<cluster>``."""
    first = output.strip().splitlines()[0] if output.strip() else ""
    job_id = first.split(";")[0].strip()
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {output!r}")
    return job_id


class SSHTransport:
    """Drive Slurm through ``sbatch``/``squeue``/``sacct``/``scancel`` over SSH.

    One connection per (host, port, username) is kept open and reused across
    calls; it is re-established transparently if it drops. Callers should
    batch job ids into a single ``status`` call rather than polling one job
    per call, so a busy connector does not turn into many login-node sessions.
    """

    def __init__(
        self,
        host: str,
        username: str,
        private_key: str,
        port: int = 22,
        known_hosts: Optional[str] = None,
        skip_host_key_verification: bool = False,
        connect_timeout: float = 30.0,
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
        self._skip_host_key_verification = skip_host_key_verification
        self._connect_timeout = connect_timeout
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
        result = await conn.run(command, check=False)
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
        # squeue is authoritative for jobs still in the system; sacct fills in the ones that finished.
        # squeue exits non-zero when none of the ids exist, which is not an error for us.
        stdout, _, _ = await self._run(f"squeue -h -j {joined} -o {_q(_SQUEUE_FORMAT)}", check=False)
        states = parse_squeue(stdout)
        missing = [j for j in ids if j not in states]
        if missing:
            stdout, _, _ = await self._run(
                f"sacct -X -n -P -j {','.join(missing)} --format={_SACCT_FORMAT}", check=False
            )
            states.update(parse_sacct(stdout))
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

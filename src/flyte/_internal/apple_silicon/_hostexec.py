"""
Host execution for Apple Silicon workloads: run a task's or app's own Flyte entrypoint natively on the Mac.

A devbox container is a Linux VM with no Metal, so code inside it can never use the Mac's GPU:
`torch.device("mps")` does not exist there. A `gpu="AppleSilicon"` workload therefore hands its entrypoint invocation
(`a0 ...` for a task, `fserve ...` for an app) to the bridge, which runs the same command on the Mac:

- in a Python environment built with `uv` for macOS from the workload's pip packages, cached per environment;
- with this host's `flyte` installed (editable when it is a source checkout);
- with the in-cluster endpoints the container was given rewritten to the devbox ports the Mac can reach.

The container streams the host process's log, mirrors its exit code, and (for apps) forwards its port to it. This trades
the container's isolation for real GPU access, so it only happens on a devbox started with
`flyte start devbox --apple-silicon`, whose ports are bound to 127.0.0.1. A workload with
`FLYTE_APPLE_SILICON_EXECUTION=container` keeps the container's isolation instead and calls a host-side MLX daemon
(`_daemon`) over an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ._daemon import DaemonError, free_port

# Set in the host process so its entrypoint runs the workload instead of delegating again.
ON_HOST_ENV = "_F_APPLE_SILICON_ON_HOST"
# Set by the SDK at serialization for a workload that runs on the host (the default for gpu="AppleSilicon").
EXECUTION_ENV = "_F_APPLE_SILICON_EXECUTION"
PACKAGES_ENV = "_F_APPLE_SILICON_HOST_PACKAGES"
# The port a host-executed app must listen on; `AppEnvironment.get_port()` reports it.
HOST_PORT_ENV = "_F_APPLE_SILICON_HOST_PORT"

HOST_EXEC_FAILED = "DAEMON_START_FAILED"

# In-cluster addresses the devbox hands its containers, and what the Mac reaches the same services at.
DEVBOX_ENDPOINT_REWRITES = {
    "rustfs-svc.flyte:9000": "localhost:30002",
    "flyte-binary-http.flyte:8090": "localhost:30080",
    "flyte-binary-grpc.flyte:8089": "localhost:30080",
}

# Container environment that describes the container itself, not the workload: never forwarded to the host.
_ENV_DROP = re.compile(
    r"^(PATH|HOME|HOSTNAME|PWD|OLDPWD|SHLVL|TERM|LANG|LC_\w+|TMPDIR|USER|SHELL|VIRTUAL_ENV|PYTHON\w*|LD_\w+|"
    r"UV_\w+|PIP_\w+|KUBERNETES_\w+|FLYTE_APPLE_SILICON_(ENDPOINT|TOKEN|WORKLOAD)|OPENAI_(BASE_URL|API_KEY)|"
    r"_F_APPLE_SILICON_\w+|SSL_CERT_\w+|REQUESTS_CA_BUNDLE)$"
    r"|_SERVICE_(HOST|PORT)\w*$|_PORT(_\d+_(TCP|UDP)\w*)?$"
)


def forwardable_env(environ: dict[str, str]) -> dict[str, str]:
    """The workload's environment, minus what only makes sense inside the container."""
    return {k: v for k, v in environ.items() if not _ENV_DROP.search(k)}


def rewrite_endpoints(env: dict[str, str], rewrites: dict[str, str] = DEVBOX_ENDPOINT_REWRITES) -> dict[str, str]:
    out = {}
    for k, value in env.items():
        for src, dst in rewrites.items():
            value = value.replace(src, dst)  # noqa: PLW2901
        out[k] = value
    return out


def _host_flyte_requirement() -> list[str]:
    """How to install this host's flyte into a workload environment: editable from a checkout, else pinned."""
    try:
        dist = importlib.metadata.distribution("flyte")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable") and direct.get("url", "").startswith("file://"):
            return ["-e", direct["url"][len("file://") :]]
        return [f"flyte=={dist.version}"]
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return ["flyte"]


class _EnvCache:
    """Builds one `uv` environment per (python, packages, flyte) and reuses it across workloads."""

    def __init__(self, root: Path):
        self.root = root
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def ensure(self, python: str, packages: list[str], log: Any) -> Path:
        flyte_req = _host_flyte_requirement()
        key = hashlib.sha256(json.dumps([python, sorted(packages), flyte_req]).encode()).hexdigest()[:16]
        env_dir = self.root / key
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if (env_dir / ".ready").exists():
                log.write(f"[flyte-apple-silicon] reusing host environment {env_dir}\n".encode())
                return env_dir
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError("host execution needs `uv` on the Mac (https://docs.astral.sh/uv/)")
            shutil.rmtree(env_dir, ignore_errors=True)
            log.write(
                f"[flyte-apple-silicon] building host environment (python {python}): "
                f"{', '.join(packages) or '-'}\n".encode()
            )
            log.flush()
            for cmd in (
                [uv, "venv", "--python", python, str(env_dir)],
                [uv, "pip", "install", "--python", str(env_dir / "bin" / "python"), *flyte_req, *packages],
            ):
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                if proc.returncode != 0:
                    raise RuntimeError(f"`{' '.join(cmd[:3])} ...` failed with exit code {proc.returncode}")
            (env_dir / ".ready").touch()
            return env_dir


class HostProcess:
    """
    A workload's entrypoint running natively on the Mac. Quacks like `_daemon.Daemon` for the bridge's
    bookkeeping (state, error, refresh, stop, read_log, snapshot).
    """

    def __init__(
        self,
        workload_id: str,
        entrypoint: str,
        argv: list[str],
        env: dict[str, str],
        python: str,
        packages: list[str],
        state_dir: Path,
        env_cache: _EnvCache,
        app_port: bool = False,
    ):
        self.workload_id = workload_id
        self.entrypoint = entrypoint
        self.argv = argv
        self.env = env
        self.python = python
        self.packages = packages
        self.model: str | None = None
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", workload_id)
        self.log_path = state_dir / "logs" / f"{safe}.log"
        self.work_dir = state_dir / "work" / safe
        self.env_cache = env_cache
        self.port: int | None = free_port() if app_port else None
        self.state = "preparing"
        self.error: DaemonError | None = None
        self.exit_code: int | None = None
        self.started_at = time.time()
        self._proc: subprocess.Popen | None = None
        self._stopped = False
        self._lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self) -> None:
        threading.Thread(target=self._run, name=f"apple-silicon-exec-{self.workload_id}", daemon=True).start()

    def _run(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        with open(self.log_path, "ab") as log:
            try:
                env_dir = self.env_cache.ensure(self.python, self.packages, log)
            except Exception as e:  # environment build failures are reported, not raised
                with self._lock:
                    self.state = "start_failed"
                    self.error = DaemonError(HOST_EXEC_FAILED, f"could not build the host environment: {e}")
                return
            env = {
                **{k: v for k, v in os.environ.items() if k in ("HOME", "USER", "LANG", "TMPDIR", "SHELL")},
                "PATH": f"{env_dir / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
                "VIRTUAL_ENV": str(env_dir),
                **self.env,
                ON_HOST_ENV: "1",
            }
            if self.port is not None:
                env[HOST_PORT_ENV] = str(self.port)
                env["PORT"] = str(self.port)
            cmd = [str(env_dir / "bin" / self.entrypoint), *self.argv]
            log.write(
                f"[flyte-apple-silicon] running on the host: {self.entrypoint} {' '.join(self.argv)[:300]}\n".encode()
            )
            log.flush()
            with self._lock:
                if self._stopped:
                    return
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=self.work_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self.state = "running"
            code = self._proc.wait()
            with self._lock:
                self.exit_code = code
                if self.state == "running":
                    self.state = "oom_killed" if code == -signal.SIGKILL else "exited"
                    if code == -signal.SIGKILL:
                        self.error = DaemonError("DAEMON_OOM", "the host process was killed (out of memory)")

    def refresh(self) -> str:
        return self.state

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            proc = self._proc
            if self.state in ("preparing", "running"):
                self.state = "torn_down"
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def read_log(self, offset: int, limit: int = 64 * 1024) -> tuple[bytes, int]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                chunk = f.read(limit)
        except FileNotFoundError:
            return b"", offset
        return chunk, offset + len(chunk)

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload": self.workload_id,
            "execution": "host",
            "entrypoint": self.entrypoint,
            "state": self.state,
            "model": None,
            "pid": self.pid,
            "port": self.port,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "error_code": self.error.code if self.error else None,
            "message": self.error.message if self.error else None,
            "log": str(self.log_path),
        }

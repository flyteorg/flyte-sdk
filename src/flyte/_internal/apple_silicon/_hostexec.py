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
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from ._daemon import DAEMON_OOM, DaemonError, free_port, kill_group, read_log_lines

# Set in the host process so its entrypoint runs the workload instead of delegating again.
ON_HOST_ENV = "_F_APPLE_SILICON_ON_HOST"
# Set by the SDK at serialization for a workload that runs on the host (the default for gpu="AppleSilicon").
EXECUTION_ENV = "_F_APPLE_SILICON_EXECUTION"
PACKAGES_ENV = "_F_APPLE_SILICON_HOST_PACKAGES"
PIP_ARGS_ENV = "_F_APPLE_SILICON_HOST_PIP_ARGS"
# The port a host-executed app must listen on; `AppEnvironment.get_port()` reports it.
HOST_PORT_ENV = "_F_APPLE_SILICON_HOST_PORT"

HOST_EXEC_FAILED = "APPLE_SILICON_HOST_EXEC_FAILED"

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
    r"_F_APPLE_SILICON_\w+|SSL_CERT_\w+|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|CUDA_\w+|NVIDIA_\w+)$"
)
# A Kubernetes service's injected variables: FOO_SERVICE_HOST, FOO_SERVICE_PORT[_NAME], FOO_PORT=tcp://...,
# FOO_PORT_443_TCP[_PROTO|_PORT|_ADDR]. Only dropped for services Kubernetes actually injected (see below).
_K8S_SERVICE_VAR = re.compile(r"^(?P<svc>[A-Z0-9_]+?)_(SERVICE_(HOST|PORT)(_\w+)?|PORT(_\d+_(TCP|UDP|SCTP)(_\w+)?)?)$")
_K8S_SERVICE_URL = re.compile(r"^(tcp|udp|sctp)://")
# Paths that only exist inside a Flyte image; a variable pointing there (HF_HOME=/root/.cache/...) would make the host
# process try to write where it can't.
_CONTAINER_ONLY_ROOTS = ("/root/", "/home/flyte/", "/opt/venv/")


def _k8s_services(environ: dict[str, str]) -> set[str]:
    """The service names Kubernetes injected variables for: those with both FOO_SERVICE_HOST and FOO_PORT=tcp://..."""
    return {
        k[: -len("_SERVICE_HOST")]
        for k in environ
        if k.endswith("_SERVICE_HOST") and _K8S_SERVICE_URL.match(environ.get(k[: -len("_SERVICE_HOST")] + "_PORT", ""))
    }


def forwardable_env(environ: dict[str, str]) -> dict[str, str]:
    """The workload's environment, minus what only makes sense inside the container."""
    services = _k8s_services(environ)
    out = {}
    for k, v in environ.items():
        if _ENV_DROP.match(k):
            continue
        m = _K8S_SERVICE_VAR.match(k)
        if m and any(k.startswith(svc + "_") for svc in services):
            continue
        if v.startswith(_CONTAINER_ONLY_ROOTS) and not Path(v).exists():
            continue
        out[k] = v
    return out


def rewrite_endpoints(env: dict[str, str], rewrites: dict[str, str] = DEVBOX_ENDPOINT_REWRITES) -> dict[str, str]:
    out = {}
    for k, value in env.items():
        for src, dst in rewrites.items():
            value = value.replace(src, dst)  # noqa: PLW2901
        out[k] = value
    return out


def safe_name(workload_id: str) -> str:
    """A file name for a workload: readable, never `.`/`..`, and distinct for distinct ids."""
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", workload_id)[:80]
    return f"{readable}-{hashlib.sha256(workload_id.encode()).hexdigest()[:10]}"


# The pip options a host environment may be built with: the image's own index settings, nothing else.
_PIP_ARG_FLAGS = ("--index-url", "--extra-index-url", "--find-links", "--prerelease", "--pre")


def validate_pip_args(args: list[str]) -> list[str]:
    for i, arg in enumerate(args):
        flag = arg.split("=", 1)[0]
        if arg.startswith("-") and flag not in _PIP_ARG_FLAGS:
            raise ValueError(f"unsupported pip option for a host environment: {arg}")
        if not arg.startswith("-") and (i == 0 or args[i - 1] not in _PIP_ARG_FLAGS):
            raise ValueError(f"unexpected pip argument for a host environment: {arg}")
    return args


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
    """Builds one `uv` environment per (python, packages, pip options, flyte) and reuses it across workloads."""

    def __init__(self, root: Path):
        self.root = root
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def ensure(self, python: str, packages: list[str], log: Any, pip_args: list[str] | None = None) -> Path:
        flyte_req = _host_flyte_requirement()
        pip_args = list(pip_args or [])
        key = hashlib.sha256(json.dumps([python, sorted(packages), pip_args, flyte_req]).encode()).hexdigest()[:16]
        env_dir = self.root / key
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:  # a fan-out of one environment builds it once; the rest wait here
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
                [uv, "pip", "install", "--python", str(env_dir / "bin" / "python"), *pip_args, *flyte_req, *packages],
            ):
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                if proc.returncode != 0:
                    raise RuntimeError(f"`{' '.join(cmd[:3])} ...` failed with exit code {proc.returncode}")
            (env_dir / ".ready").touch()
            return env_dir


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


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
        declared_port: int | None = None,
        pip_args: list[str] | None = None,
    ):
        self.workload_id = workload_id
        self.entrypoint = entrypoint
        self.argv = argv
        self.env = env
        self.python = python
        self.packages = packages
        self.pip_args = list(pip_args or [])
        self.model: str | None = None
        safe = safe_name(workload_id)
        self.log_path = state_dir / "logs" / f"{safe}.log"
        self.work_dir = state_dir / "work" / safe
        self.env_cache = env_cache
        # An app listens on the port it declared when that's free on the Mac, so apps that bind it directly (a
        # FastAPI app, a command with `--port 8080`) work; otherwise on a free one, reported through `get_port()`.
        self.port: int | None = None
        if app_port:
            self.port = declared_port if declared_port and _port_is_free(declared_port) else free_port()
        self.state = "preparing"
        self.error: DaemonError | None = None
        self.exit_code: int | None = None
        self.started_at = time.time()
        self._proc: subprocess.Popen | None = None
        self._pgid: int | None = None
        self._stopped = False
        self._lock = threading.Lock()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self) -> None:
        # A new run starts a new log before the exec request returns, so a container mirroring it can never read an
        # earlier run's lines. Owner-only, since the workload's own output can carry secrets.
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600))
        threading.Thread(target=self._run, name=f"apple-silicon-exec-{self.workload_id}", daemon=True).start()

    def _fail(self, message: str) -> None:
        with self._lock:
            if self.state == "preparing":
                self.state = "start_failed"
                self.error = DaemonError(HOST_EXEC_FAILED, message)

    def _run(self) -> None:
        try:
            self._launch()
        except Exception as e:  # anything that stops the process from starting fails the workload, never hangs it
            self._fail(f"could not start {self.entrypoint} on the host: {e}")
            return
        proc = self._proc
        if proc is None:
            return
        code = proc.wait()
        # The entrypoint's own children (an app server under `fserve`) must not outlive it.
        kill_group(self._pgid, grace=5.0)
        with self._lock:
            self.exit_code = code
            if self.state == "running":
                self.state = "oom_killed" if code == -signal.SIGKILL else "exited"
                if code == -signal.SIGKILL:
                    self.error = DaemonError(DAEMON_OOM, "the host process was killed (out of memory)")

    def _launch(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)
        self.work_dir.mkdir(parents=True)
        with open(self.log_path, "ab") as log:
            try:
                env_dir = self.env_cache.ensure(self.python, self.packages, log, self.pip_args)
            except Exception as e:  # environment build failures are reported, not raised
                self._fail(f"could not build the host environment: {e}")
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
                self._pgid = self._proc.pid  # start_new_session: the leader's pid is the group id
                self.state = "running"

    def refresh(self) -> str:
        return self.state

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            if self.state in ("preparing", "running"):
                self.state = "torn_down"
            pgid = self._pgid
        # Always the whole group: the leader may have exited and left its children running.
        kill_group(pgid, grace=10.0)

    def read_log(self, offset: int, limit: int = 64 * 1024) -> tuple[bytes, int]:
        return read_log_lines(self.log_path, offset, limit)

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

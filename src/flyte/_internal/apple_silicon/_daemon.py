"""
One supervised host-side MLX daemon (`mlx_lm.server`), serving a single Apple Silicon workload.

Lifecycle (PRD "Host MLX daemon lifecycle"):

    stopped -> starting -> serving -> torn_down
                       \\-> start_failed
    serving -> oom_killed -> torn_down
    serving -> terminated -> torn_down

A daemon that failed keeps its error code after it is torn down, so the bridge can still say why.
"""

from __future__ import annotations

import enum
import json
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DaemonState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    SERVING = "serving"
    START_FAILED = "start_failed"
    OOM_KILLED = "oom_killed"
    TERMINATED = "terminated"
    TORN_DOWN = "torn_down"


# States from which the daemon can no longer serve requests.
DEAD_STATES = frozenset(
    {DaemonState.START_FAILED, DaemonState.OOM_KILLED, DaemonState.TERMINATED, DaemonState.TORN_DOWN}
)

# Error codes, shared with flyte.errors.APPLE_SILICON_ERRORS_BY_CODE.
DAEMON_START_FAILED = "DAEMON_START_FAILED"
DAEMON_OOM = "DAEMON_OOM"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"

# Log fragments that mean the requested model could not be found or read, as opposed to any other start failure.
_MODEL_NOT_FOUND_MARKERS = (
    "RepositoryNotFoundError",
    "Repository Not Found",
    "is not a local folder and is not a valid model identifier",
    "does not appear to have a file named",
    "No such file or directory",
    "404 Client Error",
    "LocalEntryNotFoundError",
)
# Metal reports exhausted unified memory itself; the kernel's memory-pressure kill (jetsam) is a bare SIGKILL.
_OOM_MARKERS = ("Insufficient Memory", "out of memory", "kIOGPUCommandBufferCallbackErrorOutOfMemory")

_LOG_TAIL_BYTES = 4096


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http_status(url: str, timeout: float = 2.0) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


@dataclass
class DaemonError:
    code: str
    message: str


class Daemon:
    """
    A `mlx_lm.server` process bound to 127.0.0.1 on a free port, started for one workload.

    Args:
        workload_id: The workload the daemon serves; also names its log file.
        command: The command that runs `mlx_lm.server`, without `--host/--port/--model`.
        log_path: Where the daemon's stdout and stderr go.
        model: The model to load at start. When set, the daemon only reports `serving` once the model has
            loaded and answered a one-token request, so a bad model fails the start instead of the first call.
        start_timeout: Seconds to wait for `serving`, including any model download.
    """

    def __init__(
        self,
        workload_id: str,
        command: list[str],
        log_path: Path,
        model: str | None = None,
        start_timeout: float = 900.0,
    ):
        self.workload_id = workload_id
        self.command = command
        self.log_path = log_path
        self.model = model
        self.start_timeout = start_timeout
        self.port: int | None = None
        self.state = DaemonState.STOPPED
        self.error: DaemonError | None = None
        self.started_at: float | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._settled = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def start(self) -> None:
        with self._lock:
            if self.state != DaemonState.STOPPED:
                return
            self.port = free_port()
            args = [*self.command, "--host", "127.0.0.1", "--port", str(self.port)]
            if self.model:
                args += ["--model", self.model]
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "ab") as log:
                log.write(
                    f"[flyte-apple-silicon] starting MLX daemon for {self.workload_id}: {' '.join(args)}\n".encode()
                )
                log.flush()
                try:
                    # Own process group, so tearing the daemon down also stops anything it spawned.
                    self._proc = subprocess.Popen(
                        args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                    )
                except OSError as e:
                    self._fail_locked(DAEMON_START_FAILED, f"could not run {args[0]!r}: {e}")
                    return
            self.started_at = time.time()
            self.state = DaemonState.STARTING
        threading.Thread(
            target=self._await_serving, name=f"apple-silicon-start-{self.workload_id}", daemon=True
        ).start()

    def wait_settled(self, timeout: float | None = None) -> bool:
        """Block until the daemon has left `starting`. Returns False on timeout."""
        return self._settled.wait(timeout)

    def _await_serving(self) -> None:
        deadline = time.monotonic() + self.start_timeout
        while True:
            if self._exited():
                return
            status = _http_status(f"{self.base_url}/health")
            if status == 200:
                break
            if status == 503:
                # mlx_lm.server answers /health before the model loads; a failed load kills its generation
                # thread but not the process, and /health turns 503.
                self._fail_from_log("the MLX daemon could not load its model")
                return
            if time.monotonic() > deadline:
                self._fail(DAEMON_START_FAILED, f"the MLX daemon did not come up within {self.start_timeout:.0f}s")
                return
            time.sleep(0.5)

        if self.model and not self._warm_up(deadline):
            return
        with self._lock:
            if self.state == DaemonState.STARTING:
                self.state = DaemonState.SERVING
        self._settled.set()

    def _warm_up(self, deadline: float) -> bool:
        """Push one token through the model so it is loaded (and downloaded) before we report `serving`."""
        result: dict[str, Any] = {}

        def request() -> None:
            body = json.dumps({"prompt": "Hello", "max_tokens": 1}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/v1/completions", data=body, headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=max(1.0, deadline - time.monotonic())) as resp:
                    result["status"] = resp.status
            except urllib.error.HTTPError as e:
                result["status"] = e.code
                result["body"] = e.read().decode(errors="replace")
            except (urllib.error.URLError, OSError) as e:
                result["error"] = str(e)

        t = threading.Thread(target=request, daemon=True)
        t.start()
        while t.is_alive():
            if self._exited():
                return False
            if _http_status(f"{self.base_url}/health") == 503:
                self._fail_from_log(f"the MLX daemon could not load model {self.model!r}")
                return False
            if time.monotonic() > deadline:
                self._fail(DAEMON_START_FAILED, f"model {self.model!r} did not load within {self.start_timeout:.0f}s")
                return False
            t.join(0.5)
        if result.get("status") == 200:
            return True
        detail = result.get("body") or result.get("error") or f"HTTP {result.get('status')}"
        self._fail_from_log(f"model {self.model!r} failed its first request: {detail}")
        return False

    def _exited(self) -> bool:
        """Classify an exited process; returns True if it has exited."""
        if self._proc is None or self._proc.poll() is None:
            return False
        code = self._proc.returncode
        tail = self.log_tail()
        with self._lock:
            if self.state == DaemonState.STARTING:
                self._fail_locked(*self._classify_start_failure(tail, f"the MLX daemon exited with code {code}"))
            elif self.state == DaemonState.SERVING:
                if code == -signal.SIGKILL or any(m in tail for m in _OOM_MARKERS):
                    self.state = DaemonState.OOM_KILLED
                    self.error = DaemonError(
                        DAEMON_OOM, "MLX daemon was killed (out of memory); lower the model size or batch."
                    )
                else:
                    self.state = DaemonState.TERMINATED
                    self.error = DaemonError(DAEMON_START_FAILED, f"the MLX daemon exited with code {code}")
        return True

    def _classify_start_failure(self, tail: str, reason: str) -> tuple[str, str]:
        if self.model and any(m in tail for m in _MODEL_NOT_FOUND_MARKERS):
            return MODEL_NOT_FOUND, f"Model '{self.model}' is not available locally; specify a local path."
        last = next((line for line in reversed(tail.strip().splitlines()) if line.strip()), "")
        return DAEMON_START_FAILED, f"MLX daemon failed to start: {reason}" + (f" ({last.strip()})" if last else "")

    def _fail(self, code: str, message: str) -> None:
        with self._lock:
            self._fail_locked(code, message)
        self._kill()

    def _fail_from_log(self, reason: str) -> None:
        with self._lock:
            self._fail_locked(*self._classify_start_failure(self.log_tail(), reason))
        self._kill()

    def _fail_locked(self, code: str, message: str) -> None:
        self.state = DaemonState.START_FAILED
        self.error = DaemonError(code, message)
        self._settled.set()

    def refresh(self) -> DaemonState:
        """Notice a daemon that died on its own (e.g. killed under memory pressure) and return the state."""
        self._exited()
        if self.state in DEAD_STATES:
            self._settled.set()
        return self.state

    def _kill(self, grace: float = 10.0) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass

    def stop(self) -> None:
        """Tear the daemon down. A failure recorded before teardown stays in `error`."""
        self.refresh()
        self._kill()
        with self._lock:
            self.state = DaemonState.TORN_DOWN
        self._settled.set()

    def read_log(self, offset: int, limit: int = 64 * 1024) -> tuple[bytes, int]:
        """Return up to `limit` bytes of the daemon log from `offset`, and the offset to read from next."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                chunk = f.read(limit)
        except FileNotFoundError:
            return b"", offset
        return chunk, offset + len(chunk)

    def log_tail(self) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - _LOG_TAIL_BYTES))
                return f.read().decode(errors="replace")
        except FileNotFoundError:
            return ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "workload": self.workload_id,
            "state": self.state.value,
            "model": self.model,
            "pid": self.pid,
            "port": self.port,
            "started_at": self.started_at,
            "error_code": self.error.code if self.error else None,
            "message": self.error.message if self.error else None,
            "log": str(self.log_path),
        }

"""
What the `flyte` CLI on the Mac needs to run the Apple Silicon bridge for the devbox: detect Apple Silicon, find
`mlx_lm`, and start, stop and inspect the bridge process.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Kept out of ~/.flyte/devbox, which is mounted into the devbox container.
STATE_DIR = Path.home() / ".flyte" / "apple-silicon"
DEFAULT_BRIDGE_PORT = 30090
DAEMON_COMMAND_ENV = "FLYTE_APPLE_SILICON_DAEMON_COMMAND"

DAEMON_PACKAGE_MISSING = (
    "mlx-lm is not installed. Install it next to flyte (`pip install 'flyte[apple-silicon]'`) or install `uv` so the "
    f"bridge can run it with `uvx`; or set {DAEMON_COMMAND_ENV} to a command that runs `mlx_lm.server`."
)


def _is_arm64_hardware() -> bool:
    # An x86_64 Python under Rosetta reports platform.machine() == "x86_64" on Apple Silicon hardware.
    if platform.machine() == "arm64":
        return True
    try:
        out = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, check=False)
        return out.stdout.strip() == "1"
    except OSError:
        return False


def host_platform() -> str:
    """The host's os/arch as the Apple Silicon error messages quote it, e.g. `darwin/arm64`."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "darwin" and _is_arm64_hardware():
        machine = "arm64"
    return f"{system}/{machine}" + (" (Intel Mac)" if system == "darwin" and machine != "arm64" else "")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and _is_arm64_hardware()


def apple_silicon_unavailable_reason() -> str | None:
    """Why this host cannot provide the Apple Silicon accelerator, or None if it can."""
    if is_apple_silicon():
        return None
    return f"gpu='AppleSilicon' requires an Apple Silicon devbox; this host is {host_platform()}."


def resolve_daemon_command() -> list[str] | None:
    """The command that runs `mlx_lm.server` on this host, or None if there is none."""
    override = os.getenv(DAEMON_COMMAND_ENV)
    if override:
        return shlex.split(override)
    if importlib.util.find_spec("mlx_lm") is not None:
        return [sys.executable, "-m", "mlx_lm", "server"]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "mlx-lm", "mlx_lm.server"]
    return None


def load_or_create_secret(state_dir: Path = STATE_DIR) -> str:
    """
    The devbox's bridge secret. It is stable across bridge restarts because the devbox bakes it into the
    backend config at container creation: a new secret would orphan every workload of a running devbox.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "secret"
    if path.exists():
        return path.read_text().strip()
    secret = secrets.token_hex(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secret)
    return secret


@dataclass
class BridgeProcess:
    pid: int
    port: int

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _pid_file(state_dir: Path) -> Path:
    return state_dir / "bridge.json"


def _read_process(state_dir: Path) -> BridgeProcess | None:
    try:
        data = json.loads(_pid_file(state_dir).read_text())
        return BridgeProcess(pid=int(data["pid"]), port=int(data["port"]))
    except (FileNotFoundError, ValueError, KeyError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def bridge_healthy(port: int, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def running_bridge(state_dir: Path = STATE_DIR) -> BridgeProcess | None:
    proc = _read_process(state_dir)
    if proc and _alive(proc.pid) and bridge_healthy(proc.port):
        return proc
    return None


def start_bridge(
    port: int = DEFAULT_BRIDGE_PORT,
    state_dir: Path = STATE_DIR,
    timeout: float = 30.0,
) -> BridgeProcess:
    """
    Start the bridge in the background (or reuse the one already running with the same settings) and wait until
    it answers. Its workloads run their code natively on this host; see `_hostexec`.

    Raises:
        RuntimeError: when this host cannot run the bridge, or the bridge does not come up.
    """
    reason = apple_silicon_unavailable_reason()
    if reason:
        raise RuntimeError(f"APPLE_SILICON_NOT_AVAILABLE: {reason}")
    if resolve_daemon_command() is None:
        raise RuntimeError(f"DAEMON_START_FAILED: {DAEMON_PACKAGE_MISSING}")

    existing = running_bridge(state_dir)
    if existing and existing.port == port:
        return existing
    if existing:
        stop_bridge(state_dir)
    if bridge_healthy(port):
        raise RuntimeError(f"port {port} is already serving another Apple Silicon bridge; stop it first")

    load_or_create_secret(state_dir)
    log_path = state_dir / "bridge.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "flyte._internal.apple_silicon._bridge",
                "--port",
                str(port),
                "--state-dir",
                str(state_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # outlives the `flyte` command that started it
        )
    _pid_file(state_dir).write_text(json.dumps({"pid": proc.pid, "port": port}))

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace").strip().splitlines()[-5:]
            raise RuntimeError("the Apple Silicon bridge exited during startup:\n" + "\n".join(tail))
        if bridge_healthy(port):
            return BridgeProcess(pid=proc.pid, port=port)
        time.sleep(0.25)
    proc.terminate()
    raise RuntimeError(f"the Apple Silicon bridge did not answer on port {port} within {timeout:.0f}s; see {log_path}")


def stop_bridge(state_dir: Path = STATE_DIR, timeout: float = 30.0) -> bool:
    """Stop the bridge and, with it, every MLX daemon. Returns False if no bridge was running."""
    proc = _read_process(state_dir)
    _pid_file(state_dir).unlink(missing_ok=True)
    if proc is None or not _alive(proc.pid):
        return False
    os.kill(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _alive(proc.pid):
        time.sleep(0.2)
    if _alive(proc.pid):
        os.kill(proc.pid, signal.SIGKILL)
    return True


def bridge_status(state_dir: Path = STATE_DIR) -> dict[str, Any]:
    """The bridge's state and every workload's daemon, for `flyte get devbox`."""
    status: dict[str, Any] = {
        "available": apple_silicon_unavailable_reason() is None,
        "host_platform": host_platform(),
        "running": False,
        "workloads": [],
    }
    proc = running_bridge(state_dir)
    if proc is None:
        return status
    status.update(running=True, pid=proc.pid, port=proc.port, url=proc.url)
    req = urllib.request.Request(
        f"{proc.url}/apple-silicon/v1/workloads",
        headers={"Authorization": f"Bearer {load_or_create_secret(state_dir)}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status["workloads"] = json.loads(resp.read()).get("workloads", [])
    except (urllib.error.URLError, OSError, ValueError):
        pass
    return status


def devbox_container_env(port: int | None) -> dict[str, str]:
    """
    The environment `flyte start devbox` hands the devbox container, which its bootstrap renders into the
    backend's `apple-silicon-bridge` config. Without a port the backend has no bridge, and Apple Silicon workloads fail
    with an error naming this host.
    """
    env = {"FLYTE_DEVBOX_HOST_PLATFORM": host_platform()}
    if port is not None:
        env["FLYTE_DEVBOX_APPLE_SILICON_PORT"] = str(port)
        env["FLYTE_DEVBOX_APPLE_SILICON_TOKEN_SECRET"] = load_or_create_secret()
    return env

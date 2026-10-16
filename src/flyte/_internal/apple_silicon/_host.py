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

UV_MISSING = (
    "`uv` is not installed on this Mac. Apple Silicon workloads run in environments the bridge builds with it: "
    "install it from https://docs.astral.sh/uv/."
)
DAEMON_PACKAGE_MISSING = (
    "container mode (FLYTE_APPLE_SILICON_EXECUTION=container) needs mlx-lm on this Mac: install it next to flyte "
    f"(`pip install 'flyte[apple-silicon]'`), or set {DAEMON_COMMAND_ENV} to a command that runs `mlx_lm.server`."
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


def load_or_create_secret(state_dir: Path | None = None) -> str:
    """
    The devbox's bridge secret. It is stable across bridge restarts because the devbox bakes it into the
    backend config at container creation: a new secret would orphan every workload of a running devbox.
    """
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    from ._token import MIN_SECRET_LENGTH

    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "secret"
    existing = read_secret(state_dir)
    if existing is not None and len(existing) >= MIN_SECRET_LENGTH:
        return existing
    # Written whole and then linked into place, so a crash can never leave an empty secret behind, and two CLIs
    # creating it at once agree on one.
    tmp = state_dir / f".secret.{os.getpid()}.{secrets.token_hex(4)}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(32))
    try:
        if existing is not None:
            os.replace(tmp, path)  # a damaged secret is replaced, not trusted
        else:
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass  # another CLI got there first; use its secret
    finally:
        tmp.unlink(missing_ok=True)
    secret = read_secret(state_dir)
    if secret is None or len(secret) < MIN_SECRET_LENGTH:
        raise RuntimeError(f"could not create the Apple Silicon bridge secret at {path}")
    return secret


def read_secret(state_dir: Path | None = None) -> str | None:
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    try:
        return (state_dir / "secret").read_text().strip()
    except FileNotFoundError:
        return None


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
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def bridge_health(port: int, timeout: float = 2.0) -> dict[str, Any] | None:
    """What the bridge on `port` says about itself (pid, flyte version), or None if nothing healthy answers."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read() or b"{}")
            return payload if isinstance(payload, dict) else {}
    except (urllib.error.URLError, OSError, ValueError):
        return None


def bridge_healthy(port: int, timeout: float = 2.0) -> bool:
    return bridge_health(port, timeout) is not None


def _is_bridge_process(proc: BridgeProcess) -> bool:
    """Whether `proc.pid` is still our bridge, and not some other process that reused a stale pid."""
    health = bridge_health(proc.port)
    if health is not None and health.get("pid") == proc.pid:
        return True
    try:
        out = subprocess.run(["ps", "-o", "command=", "-p", str(proc.pid)], capture_output=True, text=True, check=False)
    except OSError:
        return False
    return "flyte._internal.apple_silicon._bridge" in out.stdout


def running_bridge(state_dir: Path | None = None) -> BridgeProcess | None:
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    proc = _read_process(state_dir)
    if proc and _alive(proc.pid) and bridge_healthy(proc.port) and _is_bridge_process(proc):
        return proc
    return None


def start_bridge(
    port: int = DEFAULT_BRIDGE_PORT,
    state_dir: Path | None = None,
    timeout: float = 30.0,
) -> BridgeProcess:
    """
    Start the bridge in the background (or reuse the one already running with the same settings) and wait until
    it answers. Its workloads run their code natively on this host; see `_hostexec`.

    Raises:
        RuntimeError: when this host cannot run the bridge, or the bridge does not come up.
    """
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    reason = apple_silicon_unavailable_reason()
    if reason:
        raise RuntimeError(f"APPLE_SILICON_NOT_AVAILABLE: {reason}")
    if shutil.which("uv") is None:
        raise RuntimeError(UV_MISSING)

    existing = running_bridge(state_dir)
    if existing and existing.port == port and _same_version(existing):
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
    _pid_file(state_dir).unlink(missing_ok=True)
    raise RuntimeError(f"the Apple Silicon bridge did not answer on port {port} within {timeout:.0f}s; see {log_path}")


def _same_version(proc: BridgeProcess) -> bool:
    """Whether the running bridge is this flyte's: after an upgrade, an old bridge must not keep serving."""
    from flyte._version import __version__

    health = bridge_health(proc.port) or {}
    return health.get("flyte_version") in (None, __version__)


def _signal(pid: int, sig: int) -> bool:
    try:
        os.kill(pid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_bridge(state_dir: Path | None = None, timeout: float = 30.0) -> bool:
    """
    Stop the bridge and, with it, every workload on this host. Returns False if no bridge was running. A pid file left
    behind by a bridge that's gone (a reboot, a crash) is only removed: its pid may belong to another process by now.
    """
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    proc = _read_process(state_dir)
    _pid_file(state_dir).unlink(missing_ok=True)
    if proc is None or not _alive(proc.pid) or not _is_bridge_process(proc):
        return False
    if not _signal(proc.pid, signal.SIGTERM):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _alive(proc.pid):
        time.sleep(0.2)
    if _alive(proc.pid):
        _signal(proc.pid, signal.SIGKILL)
    return True


def bridge_status(state_dir: Path | None = None) -> dict[str, Any]:
    """The bridge's state and every workload's daemon, for `flyte get devbox`."""
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
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
    secret = read_secret(state_dir)  # read-only: a status check must not create state
    if not secret:
        return status
    req = urllib.request.Request(
        f"{proc.url}/apple-silicon/v1/workloads", headers={"Authorization": f"Bearer {secret}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return status
    workloads = payload.get("workloads") if isinstance(payload, dict) else None
    status["workloads"] = [w for w in workloads if isinstance(w, dict)] if isinstance(workloads, list) else []
    return status


def devbox_container_env(port: int | None, state_dir: Path | None = None) -> dict[str, str]:
    """
    The environment `flyte start devbox` hands the devbox container, which its bootstrap renders into the
    backend's `apple-silicon-bridge` config. Without a port the backend has no bridge, and Apple Silicon workloads fail
    with an error naming this host.
    """
    state_dir = state_dir or STATE_DIR  # resolved per call, so tests can point it elsewhere
    env = {"FLYTE_DEVBOX_HOST_PLATFORM": host_platform()}
    if port is not None:
        env["FLYTE_DEVBOX_APPLE_SILICON_PORT"] = str(port)
        env["FLYTE_DEVBOX_APPLE_SILICON_TOKEN_SECRET"] = load_or_create_secret(state_dir)
    return env

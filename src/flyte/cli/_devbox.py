from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from flyte import _sentry

_CONTAINER_NAME = "flyte-devbox"
_VOLUME_NAME = "flyte-devbox"
_KUBE_DIR = Path(
    "/tmp/.kube"
)  # This path is used to store k3s kubeconfig file, we later merge it with the default kubeconfig
_KUBECONFIG_PATH = _KUBE_DIR / "kubeconfig"
_FLYTE_DEVBOX_CONFIG_DIR = Path.home() / ".flyte" / "devbox"
_PORTS = ["6443:6443", "30000:30000", "30001:30001", "30002:30002", "30003:30003", "30080:30080", "30081:30081"]
_CONSOLE_PORT = 30080
_REGISTRY_PORT = 30000
_K8S_API_PORT = 6443
_KUBE_CONTEXT = "flyte-devbox"


def _health_url(port: int | str = _CONSOLE_PORT) -> str:
    """
    Health endpoint of the devbox API server.

    Not `/readyz`: both are served by flyte-binary, but the object store's catch-all `/`
    ingress shadows `/readyz`, which then answers 403 on a perfectly healthy cluster.
    """
    return f"http://localhost:{port}/healthz"


def _docker_unavailable_reason() -> str | None:
    """Return why docker cannot be used, or None if it is usable."""
    if shutil.which("docker") is None:
        return "Docker is not installed or not on PATH. Install Docker Desktop (or the Docker Engine) and try again."
    result = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return f"Docker daemon is not running or not reachable. Start Docker and try again.\n{result.stderr.strip()}"
    return None


def _ensure_docker_available() -> None:
    reason = _docker_unavailable_reason()
    if reason:
        raise click.ClickException(reason)


def _is_kubectl_installed() -> bool:
    """Return True if kubectl is installed and on PATH, False otherwise."""
    return shutil.which("kubectl") is not None


def _run_docker(cmd: list[str], failure_message: str) -> subprocess.CompletedProcess:
    """Run a docker command and translate failure into a user-facing ClickException."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise click.ClickException(f"{failure_message}\n{details}" if details else failure_message)
    return result


def _ensure_volume(volume_name: str) -> None:
    result = _run_docker(
        ["docker", "volume", "ls", "--filter", f"name=^{volume_name}$", "--format", "{{.Name}}"],
        f"Failed to list docker volumes while checking for '{volume_name}'.",
    )
    if volume_name not in result.stdout:
        _run_docker(
            ["docker", "volume", "create", volume_name],
            f"Failed to create docker volume '{volume_name}'.",
        )


def _container_is_running(container_name: str) -> bool:
    result = _run_docker(
        ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
        f"Failed to query docker for container '{container_name}'.",
    )
    return container_name in result.stdout


def _container_is_paused(container_name: str) -> bool:
    result = _run_docker(
        [
            "docker",
            "ps",
            "--filter",
            f"name=^{container_name}$",
            "--filter",
            "status=paused",
            "--format",
            "{{.Names}}",
        ],
        f"Failed to query docker for paused container '{container_name}'.",
    )
    return container_name in result.stdout


def _resume_container(container_name: str) -> bool:
    """Unpause a container, returning False if it turned out not to be paused after all.

    Checking `status=paused` and unpausing are two separate docker calls, so the container can
    leave the paused state in between (a concurrent `docker unpause`, a daemon restart, the
    container exiting). Docker then fails with "container not paused", which is a state race
    rather than an SDK bug — report it back so the caller can fall through to the normal start
    path. Any other failure becomes a user-facing message.
    """
    result = subprocess.run(["docker", "unpause", container_name], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True
    details = (result.stderr or result.stdout or "").strip()
    if "not paused" in details.lower():
        return False
    raise click.ClickException(
        f"Failed to resume paused devbox container '{container_name}'.\n{details}"
        if details
        else f"Failed to resume paused devbox container '{container_name}'."
    )


def _is_local_image(image: str) -> bool:
    """Check if the image is local (no registry prefix)."""
    name = image.split(":", maxsplit=1)[0]
    return "/" not in name


def _pull_image(image: str) -> None:
    if _is_local_image(image):
        return
    result = subprocess.run(["docker", "pull", image], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to pull image '{image}':\n{result.stderr.strip()}")


def _run_container(
    image: str,
    is_dev_mode: bool,
    container_name: str,
    kube_dir: Path,
    flyte_devbox_config_dir: Path,
    volume_name: str,
    ports: list[str],
    gpu: bool = False,
) -> None:
    cmd = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--privileged",
        "--name",
        container_name,
        "--add-host",
        "host.docker.internal:host-gateway",
        "--env",
        f"FLYTE_DEV={'True' if is_dev_mode else 'False'}",
        "--env",
        "K3S_KUBECONFIG_OUTPUT=/.kube/kubeconfig",
        "--volume",
        f"{kube_dir.resolve()}:/.kube",
        "--volume",
        f"{flyte_devbox_config_dir}:/var/lib/flyte/config",
        "--volume",
        f"{volume_name}:/var/lib/flyte/storage",
    ]
    if gpu:
        cmd.extend(["--gpus", "all"])
    for port in ports:
        cmd.extend(["--publish", port])
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise click.ClickException(f"Failed to start container:\n{result.stderr.strip()}")


def _console_is_ready(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _wait_for_console_ready(url: str, timeout: int = 1800, poll_interval: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if _console_is_ready(url):
            return
        if time.monotonic() > deadline:
            raise click.ClickException(f"Timed out after {timeout}s waiting for Flyte cluster ({url}).")
        time.sleep(poll_interval)


def _wait_for_kubeconfig(kubeconfig_path: Path, timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout  # Set a timeout for waiting for k3s kubeconfig
    while True:
        if kubeconfig_path.exists() and kubeconfig_path.stat().st_size > 0:
            return
        if time.monotonic() > deadline:
            raise click.ClickException(f"Timed out after {timeout}s waiting for kubeconfig.")
        time.sleep(1)


def _switch_k8s_context(context: str = "flyte-devbox", namespace: str = "flyte") -> None:
    if not _is_kubectl_installed():
        console.print(
            f"[red]Warning: kubectl is not installed or not on PATH. Skipping switch to k8s context '{context}'.[/red]"
        )
        return
    try:
        subprocess.run(["kubectl", "config", "use-context", context], check=True, capture_output=True, text=True)
        subprocess.run(
            ["kubectl", "config", "set-context", "--current", f"--namespace={namespace}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = e.stderr.strip() if e.stderr else "Is kubectl installed?"
        click.echo(f"Warning: failed to switch k8s context to '{context}': {msg}", err=True)


def _flatten_kubeconfig(default_kubeconfig: Path, kubeconfig_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if default_kubeconfig.exists():
        env["KUBECONFIG"] = f"{kubeconfig_path}:{default_kubeconfig}"
    else:
        env["KUBECONFIG"] = str(kubeconfig_path)
    return subprocess.run(
        ["kubectl", "config", "view", "--flatten"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )


def _merge_kubeconfig(kubeconfig_path: Path, container_name: str) -> None:
    import tempfile

    if not _is_kubectl_installed():
        console.print(
            "[red]Warning: kubectl is not installed or not on PATH. Skipping kubeconfig merge. "
            "Install kubectl (https://kubernetes.io/docs/tasks/tools/) to interact with the devbox cluster.[/red]"
        )
        return

    default_kubeconfig = Path.home() / ".kube" / "config"
    default_kubeconfig.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = _flatten_kubeconfig(default_kubeconfig, kubeconfig_path)
    except (PermissionError, subprocess.CalledProcessError):
        # On Linux bind mounts, the in-container kubeconfig lands root-owned on
        # the host; kubectl then exits non-zero (CalledProcessError) rather than
        # Python raising PermissionError on open.
        uid, gid = os.getuid(), os.getgid()
        subprocess.run(
            ["docker", "exec", container_name, "chown", f"{uid}:{gid}", "/.kube/kubeconfig"],
            check=True,
        )
        result = _flatten_kubeconfig(default_kubeconfig, kubeconfig_path)

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".yaml") as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    shutil.move(tmp_path, default_kubeconfig)
    default_kubeconfig.chmod(0o600)


_STEPS = [
    ("Pulling image", "pull"),
    ("Starting container", "start"),
    ("Waiting for k3d cluster", "kubeconfig"),
    ("Merging kubeconfig", "merge"),
    ("Configuring kubectl context", "context"),
    ("Waiting for flyte cluster to be ready", "ready"),
]

_STEPS_DEV = _STEPS[:-1]  # Dev mode skips the readiness check

console = Console()


def _wait_for_devbox_ready(is_dev_mode: bool) -> None:
    if not is_dev_mode:
        _wait_for_console_ready(_health_url())


def stop_devbox() -> None:
    if _container_is_paused(_CONTAINER_NAME):
        console.print("[yellow]Devbox cluster is already paused.[/yellow]")
        return
    if not _container_is_running(_CONTAINER_NAME):
        console.print("[yellow]Devbox cluster is not running.[/yellow]")
        return
    _run_docker(
        ["docker", "pause", _CONTAINER_NAME],
        f"Failed to pause devbox container '{_CONTAINER_NAME}'.",
    )
    console.print("[green]Devbox cluster stopped.[/green] Run [bold]flyte start devbox[/bold] to resume.")


@_sentry.capture_errors
def launch_devbox(image_name: str, is_dev_mode: bool, gpu: bool = False, log_format: str = "console") -> None:
    _ensure_docker_available()
    _ensure_volume(_VOLUME_NAME)
    if _container_is_paused(_CONTAINER_NAME):
        console.print("[cyan]Resuming paused devbox cluster...[/cyan]")
        if _resume_container(_CONTAINER_NAME):
            return

    if _container_is_running(_CONTAINER_NAME):
        console.print("[yellow]Flyte devbox cluster is already running.[/yellow]")
        if not click.confirm("Do you want to delete the existing devbox cluster and start a new one?"):
            return
    subprocess.run(["docker", "rm", "-f", _CONTAINER_NAME], check=False, capture_output=True)

    _KUBE_DIR.mkdir(parents=True, exist_ok=True)
    # This step makes sure that we always used the latest k3s kubeconfig file
    try:
        _KUBECONFIG_PATH.unlink(missing_ok=True)
    except PermissionError as e:
        raise click.ClickException(
            f"Permission denied removing stale kubeconfig at {_KUBECONFIG_PATH}. "
            f"Delete it manually (e.g. `sudo rm {_KUBECONFIG_PATH}`) and retry.\n{e}"
        )

    steps = _STEPS_DEV if is_dev_mode else _STEPS

    if log_format == "json":
        _launch_devbox_plain(image_name, is_dev_mode, steps, gpu=gpu)
    else:
        _launch_devbox_rich(image_name, is_dev_mode, steps, gpu=gpu)


def _run_step(step_id: str, image_name: str, is_dev_mode: bool, gpu: bool = False) -> None:
    if step_id == "pull":
        _pull_image(image_name)
    elif step_id == "start":
        _run_container(
            image_name, is_dev_mode, _CONTAINER_NAME, _KUBE_DIR, _FLYTE_DEVBOX_CONFIG_DIR, _VOLUME_NAME, _PORTS, gpu=gpu
        )
    elif step_id == "kubeconfig":
        _wait_for_kubeconfig(_KUBECONFIG_PATH)
    elif step_id == "merge":
        _merge_kubeconfig(_KUBECONFIG_PATH, _CONTAINER_NAME)
    elif step_id == "context":
        _switch_k8s_context()
    elif step_id == "ready":
        _wait_for_devbox_ready(is_dev_mode)


def _launch_devbox_plain(image_name: str, is_dev_mode: bool, steps: list[tuple[str, str]], gpu: bool = False) -> None:
    for i, (description, step_id) in enumerate(steps, 1):
        click.echo(f"[{i}/{len(steps)}] {description}...")
        _run_step(step_id, image_name, is_dev_mode, gpu=gpu)
        click.echo(f"[{i}/{len(steps)}] {description}... done")

    click.echo("")
    if is_dev_mode:
        click.echo("Flyte dev cluster is running.")
    else:
        click.echo("Flyte devbox cluster is ready!")
        click.echo("  UI:             http://localhost:30080/v2")
        click.echo("  Image Registry: localhost:30000")


def _launch_devbox_rich(image_name: str, is_dev_mode: bool, steps: list[tuple[str, str]], gpu: bool = False) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[bold cyan]Starting Flyte devbox cluster", total=len(steps))

        for description, step_id in steps:
            progress.update(overall, description=f"[bold cyan]{description}")
            _run_step(step_id, image_name, is_dev_mode, gpu=gpu)
            progress.advance(overall)

    if is_dev_mode:
        console.print("[green bold]Flyte dev cluster is running.[/green bold]")
    else:
        console.print(
            Panel(
                "[green bold]Flyte devbox cluster is ready![/green bold]\n\n"
                "  🚀 UI:             [link=http://localhost:30080/v2]http://localhost:30080/v2[/link]\n"
                "  🐳 Image Registry: localhost:30000",
                title="[bold]Flyte Devbox[/bold]",
                border_style="green",
            )
        )


# Status reporting, used by `flyte get devbox`. Docker's own container states ("running",
# "paused", "exited", ...) are passed through verbatim; these two cover the rest.
_STATE_NOT_FOUND = "not-found"
_STATE_DOCKER_UNAVAILABLE = "docker-unavailable"


@dataclass
class DevboxStatus:
    """A point-in-time snapshot of the local devbox cluster."""

    state: str
    error: str | None = None
    container_id: str | None = None
    container_name: str = _CONTAINER_NAME
    created_at: str | None = None
    started_at: str | None = None
    uptime_seconds: float | None = None
    image: str | None = None
    image_id: str | None = None
    image_digest: str | None = None
    image_created: str | None = None
    image_size_bytes: int | None = None
    image_labels: dict[str, str] = field(default_factory=dict)
    dev_mode: bool = False
    gpu: bool = False
    ports: dict[str, str] = field(default_factory=dict)
    console_url: str | None = None
    endpoint: str | None = None
    registry: str | None = None
    k8s_api_url: str | None = None
    ready: bool | None = None
    cpu_percent: str | None = None
    memory_usage: str | None = None
    volume_name: str = _VOLUME_NAME
    volume_exists: bool = False
    volume_mountpoint: str | None = None
    config_dir: str = str(_FLYTE_DEVBOX_CONFIG_DIR)
    kubeconfig_path: str = str(_KUBECONFIG_PATH)
    kube_context: str | None = None
    kube_context_is_current: bool | None = None
    sdk_version: str | None = None

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _docker_json(cmd: list[str]) -> Any | None:
    """Run a docker command that emits JSON and return the parsed payload, or None on any failure."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _inspect_container(container_name: str) -> dict | None:
    payload = _docker_json(["docker", "inspect", "--type", "container", container_name])
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


def _inspect_image(image_ref: str) -> dict | None:
    payload = _docker_json(["docker", "image", "inspect", image_ref])
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


def _inspect_volume(volume_name: str) -> dict | None:
    payload = _docker_json(["docker", "volume", "inspect", volume_name])
    if isinstance(payload, list) and payload:
        return payload[0]
    return None


def _container_stats(container_name: str) -> dict | None:
    """Return a single-shot resource usage sample for the container, or None if unavailable."""
    payload = _docker_json(["docker", "stats", "--no-stream", "--format", "{{json .}}", container_name])
    return payload if isinstance(payload, dict) else None


def _parse_docker_time(value: str | None) -> datetime.datetime | None:
    """Parse a docker RFC3339 timestamp, truncating its nanoseconds to what datetime can hold."""
    if not value or value.startswith("0001-01-01"):
        return None
    text = value.replace("Z", "+00:00")
    if "." in text:
        head, _, tail = text.partition(".")
        fraction, sign, offset = (
            tail.partition("+") if "+" in tail else (tail.partition("-") if "-" in tail else (tail, "", ""))
        )
        text = f"{head}.{fraction[:6]}{sign}{offset}"
    try:
        return datetime.datetime.fromisoformat(text)
    except ValueError:
        return None


def _humanize_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def _humanize_size(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return None


def _format_timestamp(value: str | None) -> str | None:
    parsed = _parse_docker_time(value)
    if parsed is None:
        return None
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _port_bindings(inspect: dict) -> dict[str, str]:
    """Map container port (e.g. '30080/tcp') to the host port it is published on."""
    bindings: dict[str, str] = {}
    for container_port, host_bindings in (inspect.get("NetworkSettings", {}).get("Ports") or {}).items():
        if not host_bindings:
            continue
        host_port = host_bindings[0].get("HostPort")
        if host_port:
            bindings[container_port] = host_port
    return bindings


def _port_number(container_port: str) -> int:
    """Sort key for a docker port spec like '30080/tcp'."""
    try:
        return int(container_port.split("/", maxsplit=1)[0])
    except ValueError:
        return 0


def _container_env(inspect: dict) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in inspect.get("Config", {}).get("Env") or []:
        key, _, value = entry.partition("=")
        env[key] = value
    return env


def _kube_context_state(context: str = _KUBE_CONTEXT) -> tuple[str | None, bool | None]:
    """Return (context name if it exists in the kubeconfig, whether it is the current context)."""
    if not _is_kubectl_installed():
        return None, None
    contexts = subprocess.run(
        ["kubectl", "config", "get-contexts", "-o", "name"], capture_output=True, text=True, check=False
    )
    if contexts.returncode != 0 or context not in contexts.stdout.split():
        return None, None
    current = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True, check=False)
    return context, current.stdout.strip() == context


def get_devbox_status(check_ready: bool = True, check_stats: bool = True) -> DevboxStatus:
    """
    Collect a snapshot of the local devbox cluster: container state, image, endpoints and storage.

    Every probe degrades to a None/False field rather than raising, so a partially available
    environment (no kubectl, no devbox volume, a paused container) still produces a report.
    """
    from flyte._version import __version__

    docker_error = _docker_unavailable_reason()
    if docker_error:
        return DevboxStatus(state=_STATE_DOCKER_UNAVAILABLE, error=docker_error)

    volume = _inspect_volume(_VOLUME_NAME)
    kube_context, kube_context_is_current = _kube_context_state()
    status = DevboxStatus(
        state=_STATE_NOT_FOUND,
        sdk_version=__version__,
        volume_exists=volume is not None,
        volume_mountpoint=(volume or {}).get("Mountpoint"),
        kube_context=kube_context,
        kube_context_is_current=kube_context_is_current,
    )

    inspect = _inspect_container(_CONTAINER_NAME)
    if inspect is None:
        return status

    state = inspect.get("State", {})
    status.state = state.get("Status") or "unknown"
    status.container_id = (inspect.get("Id") or "")[:12] or None
    status.created_at = inspect.get("Created")
    status.started_at = state.get("StartedAt")

    started = _parse_docker_time(status.started_at)
    if started is not None and status.state in ("running", "paused"):
        status.uptime_seconds = (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()

    env = _container_env(inspect)
    status.dev_mode = env.get("FLYTE_DEV", "False").lower() == "true"
    status.gpu = bool(inspect.get("HostConfig", {}).get("DeviceRequests"))

    status.image = inspect.get("Config", {}).get("Image")
    image = _inspect_image(status.image) if status.image else None
    if image:
        status.image_id = (image.get("Id") or "").removeprefix("sha256:")[:12] or None
        digests = image.get("RepoDigests") or []
        status.image_digest = digests[0] if digests else None
        status.image_created = image.get("Created")
        status.image_size_bytes = image.get("Size")
        status.image_labels = (image.get("Config", {}) or {}).get("Labels") or {}

    status.ports = _port_bindings(inspect)
    if status.state not in ("running", "paused"):
        # A stopped container publishes nothing, so there is no endpoint to report.
        return status

    console_port = status.ports.get(f"{_CONSOLE_PORT}/tcp", str(_CONSOLE_PORT))
    registry_port = status.ports.get(f"{_REGISTRY_PORT}/tcp", str(_REGISTRY_PORT))
    k8s_port = status.ports.get(f"{_K8S_API_PORT}/tcp", str(_K8S_API_PORT))
    status.endpoint = f"localhost:{console_port}"
    status.console_url = f"http://localhost:{console_port}/v2"
    status.registry = f"localhost:{registry_port}"
    status.k8s_api_url = f"https://localhost:{k8s_port}"

    if status.state == "running":
        if check_ready:
            status.ready = _console_is_ready(_health_url(console_port))
        if check_stats:
            stats = _container_stats(_CONTAINER_NAME)
            if stats:
                status.cpu_percent = stats.get("CPUPerc")
                status.memory_usage = stats.get("MemUsage")

    return status


_STATE_DISPLAY = {
    "running": ("green", "●", "Running"),
    "paused": ("yellow", "⏸", "Paused"),
    "exited": ("red", "○", "Stopped"),
    "created": ("yellow", "○", "Created (not started)"),
    "restarting": ("yellow", "◌", "Restarting"),
    _STATE_NOT_FOUND: ("red", "○", "Not running"),
    _STATE_DOCKER_UNAVAILABLE: ("red", "✗", "Docker unavailable"),
}


def _headline(status: DevboxStatus) -> str:
    color, glyph, label = _STATE_DISPLAY.get(status.state, ("yellow", "●", status.state.capitalize()))
    parts = [f"[{color} bold]{glyph} {label}[/{color} bold]"]
    if status.ready is True:
        parts.append("[green]ready[/green]")
    elif status.ready is False:
        parts.append("[yellow]not ready yet[/yellow]")
    if status.uptime_seconds is not None:
        elapsed = _humanize_duration(status.uptime_seconds)
        # A paused container keeps its original StartedAt, so that elapsed time isn't "up".
        parts.append(f"up {elapsed}" if status.is_running else f"started {elapsed} ago")
    if status.dev_mode:
        parts.append("[cyan]dev mode[/cyan]")
    if status.gpu:
        parts.append("[magenta]GPU[/magenta]")
    return "  ·  ".join(parts)


def _section(title: str, rows: list[tuple[str, str]]) -> list[Any]:
    """Render a titled key/value block, or nothing at all when there is no row to show."""
    from flyte.cli._common import PREFERRED_ACCENT_COLOR

    if not rows:
        return []
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    return ["", f"[{PREFERRED_ACCENT_COLOR}]{title}[/{PREFERRED_ACCENT_COLOR}]", table]


def _image_label_summary(labels: dict[str, str]) -> str | None:
    """
    Summarize the OCI labels the image declares, e.g. 'k3s v1.34.6-k3s1 (rev 234e61326ca4)'.

    They describe whatever base image the devbox was built on, so they are reported as label
    metadata — the digest is what pins the running version.
    """
    title = labels.get("org.opencontainers.image.title")
    version = labels.get("org.opencontainers.image.version")
    revision = labels.get("org.opencontainers.image.revision")
    if not (version or revision):
        return None
    summary = " ".join(part for part in (title, version) if part)
    if revision:
        summary = f"{summary} (rev {revision[:12]})" if summary else f"rev {revision[:12]}"
    return summary


def render_devbox_status(status: DevboxStatus) -> Panel:
    """Render a `DevboxStatus` as a rich panel."""
    if status.state == _STATE_DOCKER_UNAVAILABLE:
        unavailable = [_headline(status), "", f"[red]{status.error}[/red]"]
        return Panel(Group(*unavailable), title="[bold]Flyte Devbox[/bold]", border_style="red", expand=False)

    body: list[Any] = [_headline(status)]

    if status.is_running:
        body += _section(
            "Endpoints",
            [
                ("UI", f"[link={status.console_url}]{status.console_url}[/link]"),
                ("API endpoint", status.endpoint or "-"),
                ("Image registry", status.registry or "-"),
                ("Kubernetes API", status.k8s_api_url or "-"),
            ],
        )

    image_rows: list[tuple[str, str]] = []
    if status.image:
        image_rows.append(("Image", status.image))
    if status.image_id:
        image_rows.append(("Image ID", status.image_id))
    if status.image_digest:
        image_rows.append(("Digest", status.image_digest))
    image_built = _format_timestamp(status.image_created)
    if image_built:
        image_rows.append(("Image built", image_built))
    size = _humanize_size(status.image_size_bytes)
    if size:
        image_rows.append(("Image size", size))
    labelled = _image_label_summary(status.image_labels)
    if labelled:
        image_rows.append(("Image labels", labelled))
    if status.sdk_version:
        image_rows.append(("Flyte SDK (local)", status.sdk_version))
    body += _section("Version", image_rows)

    container_rows: list[tuple[str, str]] = []
    if status.container_id:
        container_rows.append(("Container", f"{status.container_name} ({status.container_id})"))
    started = _format_timestamp(status.started_at)
    if started:
        container_rows.append(("Started", started))
    if status.cpu_percent:
        container_rows.append(("CPU", status.cpu_percent))
    if status.memory_usage:
        container_rows.append(("Memory", status.memory_usage))  # docker reports "<used> / <limit>"
    if status.ports:
        published = ", ".join(
            f"{host}→{container.split('/')[0]}"
            for container, host in sorted(status.ports.items(), key=lambda kv: _port_number(kv[0]))
        )
        container_rows.append(("Published ports", published))
    body += _section("Container", container_rows)

    storage_rows = [
        (
            "Docker volume",
            f"{status.volume_name} ({status.volume_mountpoint})" if status.volume_exists else "[dim]none[/dim]",
        ),
        ("Config dir", status.config_dir),
    ]
    if status.kube_context:
        current = " [green](current)[/green]" if status.kube_context_is_current else ""
        storage_rows.append(("Kube context", f"{status.kube_context}{current}"))
    body += _section("Storage & Kubernetes", storage_rows)

    body += ["", _next_step_hint(status)]

    border = "green" if status.is_running and status.ready is not False else "yellow"
    return Panel(Group(*body), title="[bold]Flyte Devbox[/bold]", border_style=border, expand=False)


def _next_step_hint(status: DevboxStatus) -> str:
    if status.state == _STATE_NOT_FOUND:
        return "Run [bold]flyte start devbox[/bold] to start a local cluster."
    if status.state == "paused":
        return "Run [bold]flyte start devbox[/bold] to resume."
    if status.state in ("exited", "created"):
        return "Run [bold]flyte start devbox[/bold] to start it again."
    if status.ready is False:
        return (
            "[yellow]The cluster is still starting up.[/yellow] Re-run [bold]flyte get devbox[/bold] "
            "in a moment, or check [bold]docker logs flyte-devbox[/bold]."
        )
    return "Point the CLI at it with [bold]flyte create config --devbox[/bold]."

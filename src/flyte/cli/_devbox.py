import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from flyte import _sentry

_CONTAINER_NAME = "flyte-devbox"
_VOLUME_NAME = "flyte-devbox"
_KUBE_DIR = Path(
    "/tmp/.kube"
)  # This path is used to store k3s kubeconfig file, we later merge it with the default kubeconfig
_KUBECONFIG_PATH = _KUBE_DIR / "kubeconfig"
_FLYTE_DEVBOX_CONFIG_DIR = Path.home() / ".flyte" / "devbox"
_PORTS = ["6443:6443", "30000:30000", "30001:30001", "30002:30002", "30003:30003", "30080:30080", "30081:30081"]
_CONSOLE_READYZ_URL = "http://localhost:30080/readyz"


def _ensure_volume(volume_name: str) -> None:
    result = subprocess.run(
        ["docker", "volume", "ls", "--filter", f"name=^{volume_name}$", "--format", "{{.Name}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if volume_name not in result.stdout:
        subprocess.run(["docker", "volume", "create", volume_name], check=True)


def _container_is_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return container_name in result.stdout


def _container_is_paused(container_name: str) -> bool:
    result = subprocess.run(
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
        capture_output=True,
        text=True,
        check=True,
    )
    return container_name in result.stdout


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
        f"{kube_dir}:/.kube",
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


def _wait_for_console_ready(url: str, timeout: int = 300, poll_interval: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
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
        _wait_for_console_ready(_CONSOLE_READYZ_URL)


def stop_devbox() -> None:
    if _container_is_paused(_CONTAINER_NAME):
        console.print("[yellow]Devbox cluster is already paused.[/yellow]")
        return
    if not _container_is_running(_CONTAINER_NAME):
        console.print("[yellow]Devbox cluster is not running.[/yellow]")
        return
    subprocess.run(["docker", "pause", _CONTAINER_NAME], check=True, capture_output=True)
    console.print("[green]Devbox cluster stopped.[/green] Run [bold]flyte start devbox[/bold] to resume.")


@_sentry.capture_errors
def launch_devbox(image_name: str, is_dev_mode: bool, gpu: bool = False, log_format: str = "console") -> None:
    _ensure_volume(_VOLUME_NAME)
    if _container_is_paused(_CONTAINER_NAME):
        console.print("[cyan]Resuming paused devbox cluster...[/cyan]")
        subprocess.run(["docker", "unpause", _CONTAINER_NAME], check=True, capture_output=True)
        return

    if _container_is_running(_CONTAINER_NAME):
        console.print("[yellow]Flyte devbox cluster is already running.[/yellow]")
        if not click.confirm("Do you want to delete the existing devbox cluster and start a new one?"):
            return
        subprocess.run(["docker", "stop", _CONTAINER_NAME], check=True, capture_output=True)

    _KUBE_DIR.mkdir(parents=True, exist_ok=True)
    # This step makes sure that we always used the latest k3s kubeconfig file
    if _KUBECONFIG_PATH.exists():
        _KUBECONFIG_PATH.unlink()

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

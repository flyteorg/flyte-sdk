import os
import shutil
import subprocess
import time
from pathlib import Path

import click

_CONTAINER_NAME = "flyte-demo"
_VOLUME_NAME = "flyte-demo"
_KUBE_DIR = Path(
    "/tmp/.kube"
)  # This path is used to store k3s kubeconfig file, we later merge it with the default kubeconfig
_KUBECONFIG_PATH = _KUBE_DIR / "kubeconfig"
_FLYTE_DEMO_CONFIG_DIR = Path.home() / ".flyte" / "demo"
_PORTS = ["6443:6443", "30000:30000", "30001:30001", "30002:30002", "30003:30003", "30080:30080"]


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


def _is_local_image(image: str) -> bool:
    """Check if the image is local (no registry prefix)."""
    name = image.split(":")[0]
    return "/" not in name


def _pull_image(image: str) -> None:
    if _is_local_image(image):
        click.echo(f"Skipping pull for local image '{image}'")
        return
    subprocess.run(["docker", "pull", image], check=True)


def _run_container(
    image: str,
    is_dev_mode: bool,
    container_name: str,
    kube_dir: Path,
    flyte_demo_config_dir: Path,
    volume_name: str,
    ports: list[str],
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
        f"{flyte_demo_config_dir}:/var/lib/flyte/config",
        "--volume",
        f"{volume_name}:/var/lib/flyte/storage",
    ]
    for port in ports:
        cmd.extend(["--publish", port])
    cmd.append(image)
    subprocess.run(cmd, check=True)


def _wait_for_kubeconfig(kubeconfig_path: Path, timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout  # Set a timeout for waiting for k3s kubeconfig
    while True:
        if kubeconfig_path.exists() and kubeconfig_path.stat().st_size > 0:
            return
        if time.monotonic() > deadline:
            raise click.ClickException(f"Timed out after {timeout}s waiting for kubeconfig.")
        time.sleep(1)


# TODO: Rename context to flyte-demo
def _switch_k8s_context(context: str = "flytev2-sandbox", namespace: str = "flyte") -> None:
    try:
        subprocess.run(["kubectl", "config", "use-context", context], check=True)
        click.echo(f"Switched k8s context to '{context}'")
        subprocess.run(["kubectl", "config", "set-context", "--current", f"--namespace={namespace}"], check=True)
        subprocess.run(
            ["kubectl", "config", "set-cluster", context, "--insecure-skip-tls-verify=true"],
            check=True,
        )
    except subprocess.CalledProcessError:
        click.echo(f"Warning: failed to switch k8s context to '{context}'. Is kubectl installed?", err=True)


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
    except PermissionError:
        # Handle the case that the user does not have permission to kubeconfig file
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
    click.echo(f"Merged kubeconfig into {default_kubeconfig}")


def launch_demo(image_name: str, is_dev_mode: bool) -> None:
    _ensure_volume(_VOLUME_NAME)

    if _container_is_running(_CONTAINER_NAME):
        click.echo(f"Container '{_CONTAINER_NAME}' is already running.")
        if not click.confirm("Do you want to delete the existing demo cluster and start a new one?"):
            return
        subprocess.run(["docker", "stop", _CONTAINER_NAME], check=True)

    _KUBE_DIR.mkdir(parents=True, exist_ok=True)
    # This step makes sure that we always used the latest k3s kubeconfig file
    if _KUBECONFIG_PATH.exists():
        _KUBECONFIG_PATH.unlink()

    _pull_image(image_name)
    _run_container(image_name, is_dev_mode, _CONTAINER_NAME, _KUBE_DIR, _FLYTE_DEMO_CONFIG_DIR, _VOLUME_NAME, _PORTS)
    _wait_for_kubeconfig(_KUBECONFIG_PATH)

    _merge_kubeconfig(_KUBECONFIG_PATH, _CONTAINER_NAME)
    _switch_k8s_context()

    click.echo("\nFlyte demo cluster is ready!")
    click.echo("UI is available at http://localhost:30080/v2")

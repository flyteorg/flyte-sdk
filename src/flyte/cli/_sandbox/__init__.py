import subprocess
from pathlib import Path

from flyte.cli._sandbox._sandbox import (
    _container_is_running,
    _ensure_volume,
    _merge_kubeconfig,
    _pull_image,
    _run_container,
    _switch_k8s_context,
    _wait_for_kubeconfig,
)

import click

_CONTAINER_NAME = "flyte-sandbox"
_VOLUME_NAME = "flyte-sandbox"
_KUBE_DIR = Path("/tmp/.kube") # This path is used to store k3s kubeconfig file, we later merge it with the default kubeconfig
_KUBECONFIG_PATH = _KUBE_DIR / "kubeconfig"
_FLYTE_SANDBOX_CONFIG_DIR = Path.home() / ".flyte" / "sandboxv2"
_PORTS = ["6443:6443", "30000:30000", "30001:30001", "30002:30002", "30003:30003", "30080:30080"]

def launch_sandbox(image_name: str, is_dev_mode: bool) -> None:
    _ensure_volume(_VOLUME_NAME)

    if _container_is_running(_CONTAINER_NAME):
        click.echo(f"Container '{_CONTAINER_NAME}' is already running.")
        if not click.confirm("Do you want to delete the existing sandbox and start a new one?"):
            return
        subprocess.run(["docker", "stop", _CONTAINER_NAME], check=True)

    _KUBE_DIR.mkdir(parents=True, exist_ok=True)
    # This step makes sure that we always used the latest k3s kubeconfig file
    if _KUBECONFIG_PATH.exists():
        _KUBECONFIG_PATH.unlink()

    _pull_image(image_name)
    _run_container(image_name, is_dev_mode, _CONTAINER_NAME, _KUBE_DIR, _FLYTE_SANDBOX_CONFIG_DIR, _VOLUME_NAME, _PORTS)
    _wait_for_kubeconfig(_KUBECONFIG_PATH)

    _merge_kubeconfig(_KUBECONFIG_PATH, _CONTAINER_NAME)
    _switch_k8s_context()


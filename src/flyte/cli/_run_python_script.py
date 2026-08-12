"""CLI command for `flyte run python-script`.

Packages a Python script and runs it on a remote Flyte cluster with
configurable resources.

Usage:

```python
flyte run python-script script.py --gpu 1 --gpu-type A100 --memory 64Gi
flyte run --follow python-script script.py --packages torch,transformers
```
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import rich_click as click

from . import _common as common
from ._common import CommandBase
from ._option import MutuallyExclusiveOption


class PythonScriptCommand(CommandBase):
    """Command that does not add --project/--domain (already on `flyte run`)."""

    common_options_enabled = False


@click.command("python-script", cls=PythonScriptCommand)
@click.argument("script", type=click.Path(exists=True, dir_okay=False))
@click.option("--cpu", type=int, default=1, show_default=True, help="Number of CPUs to request.")
@click.option("--memory", type=str, default="2Gi", show_default=True, help="Memory to request (e.g. 16Gi, 64Gi).")
@click.option("--gpu", type=int, default=0, show_default=True, help="Number of GPUs to request.")
@click.option(
    "--gpu-type",
    type=str,
    default="T4",
    show_default=True,
    help="GPU accelerator type: T4, A100, H100, L4, A10G, etc.",
)
@click.option(
    "--image",
    type=str,
    default=None,
    help="Container image URI. Mutually exclusive with --packages.",
)
@click.option(
    "--packages",
    type=str,
    default=None,
    help="Pip packages to install on the base image (comma-separated). "
    "E.g. 'torch,transformers'. Mutually exclusive with --image.",
)
@click.option("--timeout", type=int, default=3600, show_default=True, help="Task timeout in seconds.")
@click.option(
    "--extra-args",
    type=str,
    default=None,
    help="Extra arguments passed to the script (comma-separated).",
)
@click.option("--queue", type=str, default=None, help="Flyte queue / cluster override.")
@click.option(
    "--plugin-config",
    "plugin_config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["clustered"],
    help="Path to a YAML file configuring the plugin (e.g. Ray, PyTorch, Spark, Databricks) that "
    "the script runs under. The file must have a top-level 'plugin' key with the fully qualified "
    "plugin config class name (e.g. 'flyteplugins.ray.RayJobConfig') and an optional 'config' "
    "mapping with its constructor arguments, including nested classes.",
)
@click.option(
    "--clustered",
    is_flag=True,
    default=False,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Run under a ClusteredTaskEnvironment (Kubernetes JobSet) for distributed multi-node "
    "execution via torchrun, instead of a plain TaskEnvironment. Requires --replicas and "
    "--nproc-per-node.",
)
@click.option(
    "--replicas",
    type=int,
    default=None,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Number of pods (nodes) in the job set. Required with --clustered.",
)
@click.option(
    "--nproc-per-node",
    type=int,
    default=None,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Number of processes per pod, passed to `torchrun --nproc-per-node`. Required with --clustered.",
)
@click.option(
    "--rdzv-backend",
    type=click.Choice(["static", "c10d"]),
    default="static",
    show_default=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="torchrun rendezvous backend: 'static' relies on JobSet-level restarts, 'c10d' enables "
    "in-job elastic recovery. Only used with --clustered.",
)
@click.option(
    "--torchrun-max-restarts",
    type=int,
    default=0,
    show_default=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="In-pod torchrun restarts before the pod itself fails. Only used with --clustered.",
)
@click.option(
    "--cluster-max-restarts",
    type=int,
    default=0,
    show_default=True,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Number of times the entire job set may be restarted before Flyte surfaces a "
    "RetryableFailure. Only used with --clustered.",
)
@click.option(
    "--restart-on-host-maintenance",
    is_flag=True,
    default=False,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Node evictions trigger a free job set restart that doesn't consume "
    "--cluster-max-restarts. Only used with --clustered.",
)
@click.option(
    "--ttl-seconds-after-finished",
    type=int,
    default=None,
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["plugin_config"],
    help="Seconds to retain the job set after completion. Only used with --clustered.",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory path (inside the container) to upload as output after the script finishes.",
)
@click.option(
    "--include-files",
    "include_files",
    type=str,
    multiple=True,
    help="Extra paths or glob patterns (relative to the script's directory) to bundle "
    "alongside the script. Repeat the flag to pass multiple entries, "
    "e.g. --include-files '*.py' --include-files 'configs/settings.yaml'.",
)
@click.pass_obj
def python_script(
    cfg: common.CLIConfig,
    script: str,
    cpu: int,
    memory: str,
    gpu: int,
    gpu_type: str,
    image: str | None,
    packages: str | None,
    timeout: int,
    extra_args: str | None,
    queue: str | None,
    plugin_config_path: str | None,
    clustered: bool,
    replicas: int | None,
    nproc_per_node: int | None,
    rdzv_backend: Literal["static", "c10d"],
    torchrun_max_restarts: int,
    cluster_max_restarts: int,
    restart_on_host_maintenance: bool,
    ttl_seconds_after_finished: int | None,
    output_dir: str | None,
    include_files: tuple[str, ...],
) -> None:
    """Run a Python script on a remote Flyte cluster.

    Packages SCRIPT into a Flyte task, builds a container image with the
    requested resources, and submits it for remote execution.

    Project, domain, follow, and name are provided at the `flyte run` level
    (before `python-script`). Project and domain are read from the init
    config if not set.

    \b
    Examples:
        # Run a script with default resources
        flyte run python-script train.py

    \b
        # Run with GPU and extra packages
        flyte run python-script train.py --gpu 1 --gpu-type A100 --packages torch,transformers

    \b
        # Run and wait for completion (--follow is a run-level option)
        flyte run --follow python-script train.py

    \b
        # Run with a custom container image
        flyte run python-script train.py --image ghcr.io/myorg/my-image:latest

    \b
        # Run under a Ray task environment configured via YAML
        flyte run python-script train.py --plugin-config ray_config.yaml

    \b
        # ray_config.yaml:
        # plugin: flyteplugins.ray.RayJobConfig
        # config:
        #   worker_node_config:
        #     - group_name: workers
        #       replicas: 2

    \b
        # Run distributed across 4 nodes, 8 processes each, via torchrun
        flyte run python-script train.py --clustered --replicas 4 --nproc-per-node 8 --gpu 8
    """
    if image and packages:
        raise click.UsageError("--image and --packages are mutually exclusive.")
    if clustered and (replicas is None or nproc_per_node is None):
        raise click.UsageError("--clustered requires both --replicas and --nproc-per-node.")
    if not clustered and (replicas is not None or nproc_per_node is not None or ttl_seconds_after_finished is not None):
        raise click.UsageError("--replicas, --nproc-per-node, and --ttl-seconds-after-finished require --clustered.")

    from rich.console import Console

    import flyte
    from flyte._run_python_script import load_plugin_config, run_python_script
    from flyte.cli._run import initialize_config

    console = Console()

    # Read run-level options from RunArguments (set by the parent `flyte run` group)
    run_args = cfg.run_args if cfg else None
    follow = run_args.follow if run_args else False
    name = run_args.name if run_args else None
    project = run_args.project if run_args else None
    domain = run_args.domain if run_args else None
    debug = run_args.debug if run_args else False

    # Initialize flyte config (like prefetch does)
    initialize_config(
        cfg.ctx,
        project or cfg.config.task.project,
        domain or cfg.config.task.domain,
    )

    # Parse comma-separated values
    extra_args_list = [a.strip() for a in extra_args.split(",") if a.strip()] if extra_args else None

    # Build the image argument for the public API
    image_arg: flyte.Image | list[str] | None
    if image:
        if "/" in image or ":" in image:
            image_arg = flyte.Image.from_base(image)
        else:
            image_arg = flyte.Image.from_ref_name(image)
    elif packages:
        image_arg = [p.strip() for p in packages.split(",") if p.strip()]
    else:
        image_arg = None

    plugin_config = None
    if plugin_config_path:
        try:
            plugin_config = load_plugin_config(plugin_config_path)
        except ValueError as e:
            raise click.UsageError(str(e)) from e

    runtime = None
    failure_policy = None
    if clustered:
        from flyte.clustered import ClusterFailurePolicy, TorchRun

        runtime = TorchRun(rdzv_backend=rdzv_backend, max_restarts=torchrun_max_restarts)
        failure_policy = ClusterFailurePolicy(
            max_restarts=cluster_max_restarts,
            restart_on_host_maintenance=restart_on_host_maintenance,
        )

    console.print(f"[bold]Packaging script '{script}' for remote execution...[/bold]")

    run = run_python_script(
        Path(script),
        cpu=cpu,
        memory=memory,
        gpu=gpu,
        gpu_type=gpu_type,
        image=image_arg,
        timeout=timeout,
        extra_args=extra_args_list,
        queue=queue,
        wait=False,
        name=name,
        debug=debug,
        output_dir=output_dir,
        include_files=list(include_files) if include_files else None,
        plugin_config=plugin_config,
        clustered=clustered,
        replicas=replicas,
        nproc_per_node=nproc_per_node,
        runtime=runtime,
        failure_policy=failure_policy,
        ttl_seconds_after_finished=ttl_seconds_after_finished,
    )

    url = run.url
    console.print(
        f"Started run [bold]{run.name}[/bold] to execute script [bold]{script}[/bold].\n"
        f"   Check the console for status at [link={url}]{url}[/link]"
    )

    if debug:
        from flyte.cli._run import _render_debug_url

        _render_debug_url(console, run, cfg)

    if follow:
        run.wait()
        try:
            outputs = run.outputs()
            if isinstance(outputs, dict):
                result_data = outputs
            elif hasattr(outputs, "o0") and isinstance(outputs.o0, dict):
                result_data = outputs.o0
            else:
                result_data = {"exit_code": -1}
        except Exception as e:
            result_data = {"exit_code": -1}
            console.print(f"\n[bold red]Failed to get outputs:[/bold red] {e}")

        exit_code = result_data.get("exit_code", -1)
        passed = exit_code == 0

        if passed:
            console.print("\nScript completed successfully!")
        else:
            console.print(f"\nScript failed with exit code {exit_code}")
            sys.exit(1)

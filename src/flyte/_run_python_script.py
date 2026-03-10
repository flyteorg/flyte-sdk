"""Run arbitrary Python scripts on remote Flyte clusters.

Packages a Python script (or set of files) into a Flyte task and executes it
remotely with configurable resources (CPU, memory, GPU).

Public API:
    flyte.run_python_script(Path("my_script.py"), gpu=1, gpu_type="T4")
"""

# NOTE: Do NOT add ``from __future__ import annotations`` here.
# The task defined in ``_build_task`` uses ``File`` as a type annotation.
# ``typing.get_type_hints`` resolves string annotations against the
# function's ``__globals__`` (this module), so ``File`` must be a real
# class object at decoration time, not a deferred string.

import pathlib
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyte._image import Image
    from flyte.remote import Run


def _build_task(
    env: Any,
    timeout: int,
    short_name: str,
) -> Any:
    """Build the execute_script task.

    Defined separately so that the ``File`` import is evaluated eagerly
    and available as a real type when ``@env.task`` inspects annotations.
    """
    from flyte.io import File

    task_timeout = timedelta(seconds=timeout)

    @env.task(timeout=task_timeout, short_name=short_name)
    async def execute_script(script_file: File, args: list, task_timeout: int) -> dict:
        """Execute a Python script on a remote machine."""
        import subprocess
        import sys

        local_path = await script_file.download()
        cmd = [sys.executable, local_path, *args]
        result = subprocess.run(cmd, text=True, check=False, timeout=task_timeout - 60)  # noqa: ASYNC221

        return {
            "exit_code": result.returncode,
        }

    return execute_script


@syncify
async def run_python_script(
    script: pathlib.Path,
    *,
    cpu: int = 4,
    memory: str = "16Gi",
    gpu: int = 0,
    gpu_type: str = "T4",
    image: "Union[Image, List[str], None]" = None,
    timeout: int = 3600,
    extra_args: "Optional[List[str]]" = None,
    queue: "Optional[str]" = None,
    wait: bool = False,
    name: "Optional[str]" = None,
) -> "Run":
    """Package and run a Python script on a remote Flyte cluster.

    Uploads the script via :class:`~flyte.io.File`, passes it as a typed input
    to a Flyte task, and executes it remotely with the requested resources.

    Project and domain are read from the init config (set via ``flyte.init()``
    or ``flyte.init_from_config()``), consistent with ``flyte.run()``.

    :param script: Path to the Python script to run.
    :param cpu: Number of CPUs to request (default: 4).
    :param memory: Memory to request, e.g. ``"16Gi"`` (default: ``"16Gi"``).
    :param gpu: Number of GPUs to request (default: 0).
    :param gpu_type: GPU accelerator type: ``T4``, ``A100``, ``H100``, ``L4``, etc.
        Only used when ``gpu > 0`` (default: ``"T4"``).
    :param image: Container image to use. Accepts either:

        - A :class:`~flyte.Image` object for full control over the image.
        - A ``list[str]`` of pip package names to install on top of the
          default Debian base image (e.g. ``["torch", "transformers"]``).
        - ``None`` to use a plain Debian base image (default).

    :param timeout: Task timeout in seconds (default: 3600).
    :param extra_args: Extra arguments passed to the script.
    :param queue: Flyte queue / cluster override.
    :param wait: If True, block until execution completes before returning.
    :param name: Run name. If omitted, a random name is generated.
    :return: A :class:`~flyte.remote.Run` handle for the remote execution.

    Example::

        import flyte
        from pathlib import Path

        flyte.init(endpoint="my-cluster.example.com")

        # With a list of packages (auto-builds image)
        run = flyte.run_python_script(
            Path("train.py"),
            gpu=1,
            gpu_type="A100",
            memory="64Gi",
            image=["torch", "transformers"],
        )
        print(run.url)

        # With a custom Image object
        img = flyte.Image.from_debian_base(name="my-img").with_pip_packages("numpy")
        run = flyte.run_python_script(Path("analysis.py"), image=img)
    """
    import flyte
    from flyte.io import File

    script = pathlib.Path(script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    if not script.suffix == ".py":
        raise ValueError(f"Script must be a .py file, got: {script}")

    # Build image
    img: Any
    if image is None:
        img = flyte.Image.from_debian_base(name="python-script-runner")
    elif isinstance(image, list):
        img = flyte.Image.from_debian_base(name="python-script-runner").with_pip_packages(*image)
    else:
        img = image

    # Build resources
    resource_kwargs: Dict[str, Any] = {"cpu": cpu, "memory": memory}
    if gpu > 0:
        resource_kwargs["gpu"] = f"{gpu_type}:{gpu}"
    resources = flyte.Resources(**resource_kwargs)

    # Create environment
    env_kwargs: Dict[str, Any] = {
        "name": "python_script",
        "image": img,
        "resources": resources,
    }
    if queue:
        env_kwargs["queue"] = queue
    env = flyte.TaskEnvironment(**env_kwargs)

    # Build task (in a separate function so File annotation resolves correctly)
    task_short_name = name or script.stem
    execute_script = _build_task(env, timeout, short_name=task_short_name)

    script_file: File = await File.from_local(script)

    runner = flyte.with_runcontext(
        mode="remote",
        name=name,
        interactive_mode=True,
    )
    run = await runner.run.aio(
        execute_script,
        script_file=script_file,
        args=extra_args or [],
        task_timeout=timeout,
    )

    if wait:
        await run.wait.aio(quiet=True)

    return run

"""Run arbitrary Python scripts on remote Flyte clusters.

Packages a Python script (or set of files) into a Flyte task and executes it
remotely with configurable resources (CPU, memory, GPU).

Public API:
    flyte.run_python_script(Path("my_script.py"), gpu=1, gpu_type="T4")
"""

# All annotations are deferred (PEP 563) so we can keep ``flyte.io`` out of the
# ``import flyte`` critical path. ``flyte.io`` would otherwise drag the heavy
# DataFrame transformer (mashumaro.jsonschema, markdown_it, pendulum) for ~1s on
# a 1-CPU cluster cold start. ``flyte`` is imported here only as a partial
# module reference so ``get_type_hints(PythonScriptOutput)`` can resolve
# ``flyte.io.Dir`` once the inner ``_build_task`` has actually loaded
# ``flyte.io`` on demand.
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import flyte  # circular: returns the partial module; sufficient for annotation resolution.
from flyte.syncify import syncify

if TYPE_CHECKING:
    from flyte._image import Image
    from flyte.io import Dir
    from flyte.remote import Run


@dataclass
class PythonScriptOutput:
    exit_code: int
    stdout: str
    # Always populated. When the script did not request / produce an output directory this is a
    # ``flyte.io.EmptyDir()`` sentinel — check ``output_dir.is_empty`` to detect that case.
    # We avoid ``Optional[Dir]`` because Flyte/mashumaro's DataclassTransformer strips the
    # ``Optional`` wrapper around ``SerializableType`` fields and calls ``Dir._deserialize(None)``,
    # which fails with ``Field "output_dir" of type Dir in PythonScriptOutput has invalid value None``.
    output_dir: flyte.io.Dir


def _build_task(
    env: Any,
    script_name: str,
    timeout: int,
    short_name: str,
    output_dir: "Optional[str]" = None,
    task_resolver: Any = None,
) -> Any:
    """Build the `execute_script` task for serialization.

    The *script_name* is captured via closure for local execution.  When
    running remotely the `InternalTaskResolver` recreates the task from
    the loader args embedded in the container command, so the closure value
    is not carried over the wire.
    """
    task_timeout = timedelta(seconds=timeout)

    @env.task(timeout=task_timeout, short_name=short_name, task_resolver=task_resolver)
    async def execute_script(args: list[str], task_timeout: int) -> PythonScriptOutput:
        """Execute a Python script on a remote machine."""
        import os
        import subprocess
        import sys
        import tempfile

        tail_bytes = 1000
        cmd = [sys.executable, script_name, *args]

        with tempfile.TemporaryFile(mode="w+") as out_f, tempfile.TemporaryFile(mode="w+") as err_f:
            result = subprocess.run(  # noqa: ASYNC221
                cmd,
                stdout=out_f,
                stderr=err_f,
                check=False,
                timeout=task_timeout - 60,
            )

            for f in (out_f, err_f):
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                f.seek(max(0, pos - tail_bytes))

            stdout_tail = out_f.read()
            stderr_tail = err_f.read()

        if result.returncode != 0:
            raise RuntimeError(f"Script failed with exit code {result.returncode}, stderr: {stderr_tail}")

        from flyte.io import Dir, EmptyDir

        if output_dir:
            _dir: Dir = await Dir.from_local(output_dir)
        else:
            _dir = EmptyDir()

        return PythonScriptOutput(
            exit_code=result.returncode,
            stdout=stdout_tail,
            output_dir=_dir,
        )

    return execute_script


def _build_script_runner_task(script_name: str, output_dir: "Optional[str]" = None, timeout: str = "3600") -> Any:
    """Build the `execute_script` task at runtime (called by `InternalTaskResolver`).

    Creates a minimal `flyte.TaskEnvironment` — only the function
    signature matters here because the container already has the correct
    image and resources.
    """
    import flyte

    env = flyte.TaskEnvironment(name="python_script")
    return _build_task(env, script_name, int(timeout), short_name=script_name, output_dir=output_dir)


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
    debug: bool = False,
    output_dir: "Optional[str]" = None,
) -> "Run":
    """Package and run a Python script on a remote Flyte cluster.

    Bundles the script into a Flyte code bundle and executes it remotely
    with the requested resources.  Unlike `interactive_mode` (which
    pickles the task), this approach uses an `InternalTaskResolver`
    so the task can be properly debugged with `debug=True`.

    Project and domain are read from the init config (set via `flyte.init()`
    or `flyte.init_from_config()`), consistent with `flyte.run()`.

    :param script: Path to the Python script to run.
    :param cpu: Number of CPUs to request (default: 4).
    :param memory: Memory to request, e.g. `"16Gi"` (default: `"16Gi"`).
    :param gpu: Number of GPUs to request (default: 0).
    :param gpu_type: GPU accelerator type: `T4`, `A100`, `H100`, `L4`, etc.
        Only used when `gpu > 0` (default: `"T4"`).
    :param image: Container image to use. Accepts either:

        - A `flyte.Image` object for full control over the image.
        - A `list[str]` of pip package names to install on top of the
          default Debian base image (e.g. `["torch", "transformers"]`).
        - `None` to use a plain Debian base image (default).

    :param timeout: Task timeout in seconds (default: 3600).
    :param extra_args: Extra arguments passed to the script.
    :param queue: Flyte queue / cluster override.
    :param wait: If True, block until execution completes before returning.
    :param name: Run name. If omitted, a random name is generated.
    :param debug: If True, run the task as a VS Code debug task, starting a
        code-server in the container so you can connect via the UI to
        interactively debug/run the task.
    :return: A `flyte.remote.Run` handle for the remote execution.

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
    from flyte._internal.resolvers.internal import InternalTaskResolver
    from flyte._run import _Runner

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

    # Build task with the InternalTaskResolver so the runner knows how to
    # serialize and reload it without pickling.
    resolver = InternalTaskResolver(
        "flyte._run_python_script._build_script_runner_task",
        script_name=script.name,
        output_dir=output_dir,
        timeout=timeout,
    )
    task_short_name = name or script.stem
    execute_script = _build_task(
        env, script.name, timeout, short_name=task_short_name, output_dir=output_dir, task_resolver=resolver
    )

    runner = _Runner(
        force_mode="remote",
        name=name,
        debug=debug,
        copy_style="custom",
        _bundle_relative_paths=(script.name,),
        _bundle_from_dir=script.parent,
    )
    run = await runner.run.aio(
        execute_script,
        args=extra_args or [],
        task_timeout=timeout,
    )

    if wait:
        await run.wait.aio(quiet=True)

    return run

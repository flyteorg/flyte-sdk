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

import dataclasses
import json
import pathlib
import typing
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import flyte  # circular: returns the partial module; sufficient for annotation resolution.
from flyte.syncify import syncify

if TYPE_CHECKING:
    import flyte.io
    from flyte._image import Image
    from flyte.clustered import ClusterFailurePolicy, TorchRun
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


def _resolve_plugin_config_class(qualified_name: str) -> Any:
    """Dynamically import a plugin config class by its fully qualified name.

    E.g. `"flyteplugins.ray.RayJobConfig"`.
    """
    from flyte._internal.runtime.entrypoints import load_class

    try:
        return load_class(qualified_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ValueError(
            f"Could not load plugin config class {qualified_name!r}: {e}. Make sure the plugin "
            "package is installed and the name is fully qualified, "
            "e.g. 'flyteplugins.ray.RayJobConfig'."
        ) from e


def _coerce_plugin_config_value(field_type: Any, value: Any) -> Any:
    """Coerce a YAML/JSON-parsed value into the shape a dataclass field expects.

    Recurses into nested dataclasses, lists, dicts, and Optional/Union wrappers so
    plugin config classes that nest other dataclasses (e.g. `RayJobConfig.worker_node_config:
    List[WorkerNodeConfig]`) can be constructed from plain dicts/lists parsed out of YAML.
    """
    if value is None or field_type is None:
        return value

    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        # Optional[X] is Union[X, None]; try each non-None member until one sticks.
        for candidate in typing.get_args(field_type):
            if candidate is type(None):
                continue
            try:
                return _coerce_plugin_config_value(candidate, value)
            except (TypeError, ValueError):
                continue
        return value
    if origin is list and isinstance(value, list):
        args = typing.get_args(field_type)
        item_type = args[0] if args else None
        return [_coerce_plugin_config_value(item_type, item) for item in value]
    if origin is dict and isinstance(value, dict):
        args = typing.get_args(field_type)
        value_type = args[1] if len(args) > 1 else None
        return {k: _coerce_plugin_config_value(value_type, v) for k, v in value.items()}
    if isinstance(field_type, type) and dataclasses.is_dataclass(field_type) and isinstance(value, dict):
        return _build_dataclass_from_dict(field_type, value)
    return value


def _build_dataclass_from_dict(cls: type, data: "Dict[str, Any]") -> Any:
    """Recursively construct a dataclass instance (including nested dataclass fields) from a plain dict.

    This is how YAML-parsed plugin configuration becomes the dataclass instances that
    `TaskEnvironment(plugin_config=...)` expects.
    """
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls!r} is not a dataclass; cannot build it from a plugin config file.")

    hints = typing.get_type_hints(cls)
    valid_fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - valid_fields)
    if unknown:
        raise ValueError(
            f"Unknown field(s) {unknown} for plugin config class {cls.__module__}.{cls.__qualname__}. "
            f"Valid fields: {sorted(valid_fields)}"
        )

    kwargs = {key: _coerce_plugin_config_value(hints.get(key), value) for key, value in data.items()}
    return cls(**kwargs)


def load_plugin_config(path: "Union[str, pathlib.Path]") -> Any:
    """Load a plugin config instance from a YAML file.

    The file must be a mapping with a top-level `plugin` key holding the fully qualified
    class name of the plugin config (e.g. `flyteplugins.ray.RayJobConfig`), and an optional
    `config` mapping with the constructor arguments — including nested classes, expressed as
    nested mappings/lists that mirror the plugin config's dataclass fields.

    Example:

    ```yaml
    plugin: flyteplugins.ray.RayJobConfig
    config:
      worker_node_config:
        - group_name: workers
          replicas: 2
      head_node_config:
        ray_start_params:
          num-cpus: "0"
    ```
    """
    import yaml

    path = pathlib.Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict) or "plugin" not in raw:
        raise ValueError(
            f"Plugin config file {path} must be a YAML mapping with a top-level 'plugin' key "
            "(the fully qualified plugin config class name), e.g.:\n\n"
            "plugin: flyteplugins.ray.RayJobConfig\nconfig:\n  worker_node_config: [...]"
        )

    plugin_cls = _resolve_plugin_config_class(raw["plugin"])
    config = raw.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"Plugin config file {path}: 'config' must be a mapping, got {type(config)}")

    return _build_dataclass_from_dict(plugin_cls, config)


def _plugin_config_qualname(instance: Any) -> str:
    cls = type(instance)
    return f"{cls.__module__}.{cls.__qualname__}"


def _serialize_plugin_config(instance: Any) -> str:
    return json.dumps(dataclasses.asdict(instance))


def _deserialize_plugin_config(qualified_name: str, data: str) -> Any:
    plugin_cls = _resolve_plugin_config_class(qualified_name)
    return _build_dataclass_from_dict(plugin_cls, json.loads(data))


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
        import collections
        import subprocess
        import sys

        # `-u` forces line-buffered Python so prints flush into the pipe
        # immediately, giving us live streaming to the pod's stdout (and
        # therefore the k8s log stream / Flyte UI logs tab).
        cmd = [sys.executable, "-u", script_name, *args]
        tail: "collections.deque[str]" = collections.deque(maxlen=80)

        proc = subprocess.Popen(  # noqa: ASYNC220
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # unified ordering with stdout
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line)
            proc.wait(timeout=task_timeout - 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise

        stdout_tail = "".join(tail)

        if proc.returncode != 0:
            raise RuntimeError(f"Script failed with exit code {proc.returncode}, last output: {stdout_tail}")

        from flyte.io import Dir, EmptyDir

        if output_dir:
            _dir: Dir = await Dir.from_local(output_dir)
        else:
            _dir = EmptyDir()

        return PythonScriptOutput(
            exit_code=proc.returncode,
            stdout=stdout_tail,
            output_dir=_dir,
        )

    return execute_script


def _build_script_runner_task(
    script_name: str,
    output_dir: "Optional[str]" = None,
    timeout: str = "3600",
    plugin_config_class: "Optional[str]" = None,
    plugin_config_data: "Optional[str]" = None,
    clustered: "Optional[str]" = None,
) -> Any:
    """Build the `execute_script` task at runtime (called by `InternalTaskResolver`).

    Creates a minimal `flyte.TaskEnvironment` — only the function signature and
    `plugin_config` matter here: image/resources are already baked into the running
    container, but `plugin_config` is re-hydrated because task types like Ray/Spark
    read `self.plugin_config` from inside `pre()`/`execute()`, which run in this
    container at task execution time. `clustered` (set when `run_python_script` used
    `clustered=True`) is reconstructed the same way, for `TaskPluginRegistry` to route
    to `ClusteredTaskTemplate`; the actual replica/torchrun settings only matter at
    serialization time on the client and are not needed again here.
    """
    import flyte

    plugin_config: Any = None
    if plugin_config_class and plugin_config_data:
        plugin_config = _deserialize_plugin_config(plugin_config_class, plugin_config_data)
    elif clustered:
        from flyte.clustered._task import _ClusteredPlugin

        plugin_config = _ClusteredPlugin()

    env = flyte.TaskEnvironment(name="python_script", plugin_config=plugin_config)
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
    include_files: "Optional[List[str]]" = None,
    plugin_config: "Optional[Any]" = None,
    clustered: bool = False,
    replicas: "Optional[int]" = None,
    nproc_per_node: "Optional[int]" = None,
    runtime: "Optional[TorchRun]" = None,
    failure_policy: "Optional[ClusterFailurePolicy]" = None,
    ttl_seconds_after_finished: "Optional[int]" = None,
) -> "Run":
    """Package and run a Python script on a remote Flyte cluster.

    Bundles the script into a Flyte code bundle and executes it remotely
    with the requested resources.  Unlike `interactive_mode` (which
    pickles the task), this approach uses an `InternalTaskResolver`
    so the task can be properly debugged with `debug=True`.

    Project and domain are read from the init config (set via `flyte.init()`
    or `flyte.init_from_config()`), consistent with `flyte.run()`.

    Args:
        script: Path to the Python script to run.
        cpu: Number of CPUs to request (default: 4).
        memory: Memory to request, e.g. `"16Gi"` (default: `"16Gi"`).
        gpu: Number of GPUs to request (default: 0).
        gpu_type: GPU accelerator type: `T4`, `A100`, `H100`, `L4`, etc.
            Only used when `gpu > 0` (default: `"T4"`).
        image: Container image to use. Accepts either:

            - A `flyte.Image` object for full control over the image.
            - A `list[str]` of pip package names to install on top of the
              default Debian base image (e.g. `["torch", "transformers"]`).
            - `None` to use a plain Debian base image (default).
        timeout: Task timeout in seconds (default: 3600).
        extra_args: Extra arguments passed to the script.
        queue: Flyte queue / cluster override.
        wait: If True, block until execution completes before returning.
        name: Run name. If omitted, a random name is generated.
        debug: If True, run the task as a VS Code debug task, starting a
            code-server in the container so you can connect via the UI to
            interactively debug/run the task.
        include_files: Extra paths or glob patterns to bundle alongside
            the script. Relative entries anchor at the script's directory;
            absolute paths pass through unchanged. Example:
            `["*.py", "configs/settings.yaml"]`.
        plugin_config: A plugin config instance (e.g. `flyteplugins.ray.RayJobConfig`)
            that selects and configures the underlying task type the script runs
            under. Use `flyte.load_plugin_config()` to build one from a YAML file
            (this is what the `flyte run python-script --plugin-config` CLI flag
            does under the hood). Mutually exclusive with `clustered=True`, which
            manages its own plugin config.
        clustered: If True, run the script under a `flyte.clustered.ClusteredTaskEnvironment`
            (a Kubernetes JobSet) instead of a plain `TaskEnvironment`, for distributed
            multi-node execution via `torchrun`. Requires `replicas` and `nproc_per_node`.
        replicas: Number of pods (== nodes) in the job set. Required when `clustered=True`.
        nproc_per_node: Number of processes per pod, passed to `torchrun --nproc-per-node`.
            Required when `clustered=True`.
        runtime: Launcher configuration for clustered execution, e.g.
            `flyte.clustered.TorchRun(rdzv_backend="c10d")`. Only used when `clustered=True`;
            defaults to `TorchRun()`.
        failure_policy: JobSet-level restart/eviction policy, e.g.
            `flyte.clustered.ClusterFailurePolicy(max_restarts=2)`. Only used when
            `clustered=True`; defaults to `ClusterFailurePolicy()`.
        ttl_seconds_after_finished: Seconds to retain the JobSet after completion. Only
            used when `clustered=True`.

    Returns:
        A `flyte.remote.Run` handle for the remote execution.

    Example:

    ```python
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
    ```
    """
    import flyte
    from flyte._internal.resolvers.internal import InternalTaskResolver
    from flyte._run import _Runner

    script = pathlib.Path(script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    if not script.suffix == ".py":
        raise ValueError(f"Script must be a .py file, got: {script}")

    if clustered:
        if plugin_config is not None:
            raise ValueError(
                "plugin_config cannot be combined with clustered=True: ClusteredTaskEnvironment "
                "manages its own plugin config internally."
            )
        if replicas is None or nproc_per_node is None:
            raise ValueError("clustered=True requires both replicas and nproc_per_node to be set.")
    elif any(v is not None for v in (replicas, nproc_per_node, runtime, failure_policy, ttl_seconds_after_finished)):
        raise ValueError(
            "replicas, nproc_per_node, runtime, failure_policy, and ttl_seconds_after_finished "
            "only apply when clustered=True."
        )

    # Build image
    img: Any
    if image is None:
        img = flyte.Image.from_debian_base(name="python-script-runner")
    elif isinstance(image, list):
        img = flyte.Image.from_debian_base(name="python-script-runner").with_pip_packages(*cast("List[str]", image))
    else:
        img = image

    # Build resources
    resource_kwargs: Dict[str, Any] = {"cpu": cpu, "memory": memory}
    if gpu > 0:
        resource_kwargs["gpu"] = f"{gpu_type}:{gpu}"
    resources = flyte.Resources(**resource_kwargs)

    # Create environment
    env_kwargs: Dict[str, Any] = {
        "name": f"python_script_{script.stem}",
        "image": img,
        "resources": resources,
    }
    if queue:
        env_kwargs["queue"] = queue
    if include_files:
        env_kwargs["include"] = tuple(include_files)

    env: Any
    if clustered:
        import flyte.clustered

        env_kwargs["replicas"] = replicas
        env_kwargs["nproc_per_node"] = nproc_per_node
        if runtime is not None:
            env_kwargs["runtime"] = runtime
        if failure_policy is not None:
            env_kwargs["failure_policy"] = failure_policy
        if ttl_seconds_after_finished is not None:
            env_kwargs["ttl_seconds_after_finished"] = ttl_seconds_after_finished
        env = flyte.clustered.ClusteredTaskEnvironment(**env_kwargs)
    else:
        if plugin_config is not None:
            env_kwargs["plugin_config"] = plugin_config
        env = flyte.TaskEnvironment(**env_kwargs)
    # Anchor relative `include` entries at the script's directory. The default
    # stack-walk in `_get_declaring_file` lands on CLI internals, which would
    # resolve globs against the wrong anchor.
    env._declaring_file = str(script)

    # Build task with the InternalTaskResolver so the runner knows how to
    # serialize and reload it without pickling.
    resolver = InternalTaskResolver(
        "flyte._run_python_script._build_script_runner_task",
        script_name=script.name,
        output_dir=output_dir,
        timeout=timeout,
        plugin_config_class=_plugin_config_qualname(plugin_config) if plugin_config is not None else None,
        plugin_config_data=_serialize_plugin_config(plugin_config) if plugin_config is not None else None,
        clustered="1" if clustered else None,
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
    run = cast(
        "Run",
        await runner.run.aio(
            execute_script,
            args=extra_args or [],
            task_timeout=timeout,
        ),
    )

    if wait:
        await run.wait.aio(quiet=True)

    return run

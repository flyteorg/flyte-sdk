"""Persist and restore the task execution context for interactive debugging.

When a task pod enters debug mode, the runtime writes the parameters it was invoked with
(inputs path, outputs path, raw data prefix, checkpoint paths, action identity, ...) to a
well-known JSON file — the same information encoded into the generated `.vscode/launch.json`.

`flyte.load_context()` reads that file back, initializes the SDK against the cluster, and
installs a `flyte.models.TaskContext` as the current context, so code executed from a
debugger, REPL, or notebook attached to the pod behaves as if it were running inside the
task (e.g. `flyte.ctx()` works and File/Dir IO uses the task's raw data prefix).
"""

from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional

from flyte._logging import logger

if TYPE_CHECKING:
    from flyte.models import TaskContext

#: Location of the context config file, relative to the task's working directory.
CONTEXT_CONFIG_SUBPATH = pathlib.Path(".flyte") / "config.json"

#: a0 runtime parameters persisted into the config file.
_CONFIG_KEYS = (
    "inputs",
    "outputs_path",
    "version",
    "run_base_dir",
    "raw_data_path",
    "checkpoint_path",
    "prev_checkpoint",
    "name",
    "run_name",
    "run_start_time",
    "project",
    "domain",
    "org",
    "image_cache",
    "tgz",
    "pkl",
    "dest",
    "resolver",
)


def context_config_path(base_dir: str | os.PathLike | None = None) -> pathlib.Path:
    """
    The well-known path of the context config file: `<base_dir>/.flyte/config.json`,
    where base_dir defaults to the current working directory.
    """
    base = pathlib.Path(base_dir) if base_dir is not None else pathlib.Path.cwd()
    return base / CONTEXT_CONFIG_SUBPATH


def write_context_config(params: Mapping[str, Any], base_dir: str | os.PathLike | None = None) -> pathlib.Path:
    """
    Write the task runtime parameters to the well-known context config file so that
    `flyte.load_context` can restore the task context later.

    Args:
        params: The a0 entrypoint parameters (e.g. `click.Context.params`). Values for
            `run_name` / `name` should already be resolved (no `{{...}}` templates).
        base_dir: Directory under which `.flyte/config.json` is written. Defaults to cwd.

    Returns:
        The path of the written config file.
    """
    path = context_config_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"config_version": 1}
    config.update({k: params.get(k) for k in _CONFIG_KEYS})
    config["resolver_args"] = list(params.get("resolver_args") or [])
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Wrote task context config to {path}")
    return path


def _parse_run_start_time(value: str | None) -> Optional[datetime]:
    """Best-effort ISO-8601 parse; returns None for missing/unsubstituted/unparsable values."""
    if not value or value.startswith("{{"):
        return None
    raw = value.rstrip()
    # tolerate trailing "Z" — datetime.fromisoformat only handles it on 3.11+
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning(f"Could not parse run_start_time {value!r} from context config; ignoring.")
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def load_context(path: str | os.PathLike | None = None) -> TaskContext:
    """
    Restore the task execution context from the config file written by a debug-mode task pod.

    Use this from a debugger, REPL, or notebook attached to a live task pod: it initializes
    the SDK against the cluster and installs the task's context as the current one, so
    `flyte.ctx()` and data IO (raw data prefix, inputs/outputs paths) behave as they would
    inside the task.

    Args:
        path: Optional explicit path to the config file. Defaults to the well-known
            location `<cwd>/.flyte/config.json`.

    Returns:
        The restored and installed `flyte.models.TaskContext`.

    Raises:
        FileNotFoundError: If the config file does not exist at the given/known location.
        ValueError: If the config file is missing required fields.
    """
    cfg_path = pathlib.Path(path) if path is not None else context_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"No task context config found at {cfg_path}. This file is written by a task pod running in "
            "debug mode (e.g. `flyte run --debug`). Make sure you are running from the task's working "
            "directory, or pass the path explicitly: flyte.load_context(path=...)."
        )

    with open(cfg_path) as f:
        config = json.load(f)

    # run_name/name are resolved by the writer, but guard against unsubstituted templates anyway.
    run_name = config.get("run_name") or ""
    name = config.get("name") or ""
    if run_name.startswith("{{"):
        run_name = os.getenv("RUN_NAME", "")
    if name.startswith("{{"):
        name = os.getenv("ACTION_NAME", "")

    missing = [
        key
        for key, value in {
            "name": name,
            "run_name": run_name,
            "project": config.get("project"),
            "domain": config.get("domain"),
            "org": config.get("org"),
            "version": config.get("version"),
            "outputs_path": config.get("outputs_path"),
            "run_base_dir": config.get("run_base_dir"),
            "raw_data_path": config.get("raw_data_path"),
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Task context config {cfg_path} is missing required fields: {', '.join(missing)}")

    import flyte.report
    import flyte.storage as storage
    from flyte._context import internal_ctx, root_context_var
    from flyte._initialize import init_in_cluster
    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte.models import ActionID, CheckpointPaths, CodeBundle, PathRewrite, RawDataPath, TaskContext

    init_in_cluster(org=config["org"], project=config["project"], domain=config["domain"])

    # Mirror the a0 entrypoint: apply the accelerated-datasets path rewrite only if its mount exists.
    path_rewrite = None
    path_rewrite_cfg = os.getenv("_F_PATH_REWRITE")
    if path_rewrite_cfg:
        potential_path_rewrite = PathRewrite.from_str(path_rewrite_cfg)
        if storage.exists_sync(potential_path_rewrite.new_prefix):
            path_rewrite = potential_path_rewrite

    bundle = None
    if config.get("tgz") or config.get("pkl"):
        bundle = CodeBundle(
            tgz=config.get("tgz"),
            pkl=config.get("pkl"),
            destination=config.get("dest") or ".",
            computed_version=config["version"],
        )

    action = ActionID(
        name=name,
        run_name=run_name,
        project=config["project"],
        domain=config["domain"],
        org=config["org"],
    )
    tctx_kwargs: dict[str, Any] = {
        "action": action,
        "version": config["version"],
        "raw_data_path": RawDataPath(path=config["raw_data_path"], path_rewrite=path_rewrite),
        "input_path": config.get("inputs"),
        "output_path": config["outputs_path"],
        "run_base_dir": config["run_base_dir"],
        "checkpoint_paths": CheckpointPaths(
            prev_checkpoint_path=config.get("prev_checkpoint"),
            checkpoint_path=config.get("checkpoint_path"),
        ),
        "code_bundle": bundle,
        "compiled_image_cache": ImageCache.from_transport(config["image_cache"]) if config.get("image_cache") else None,
        "report": flyte.report.Report(name=name),
        "mode": "remote",
        "interactive_mode": True,
    }
    run_start_time = _parse_run_start_time(config.get("run_start_time"))
    if run_start_time is not None:
        tctx_kwargs["run_start_time"] = run_start_time
    tctx = TaskContext(**tctx_kwargs)

    # Install permanently (no context-manager scoping): a debugger/REPL session has no
    # natural exit point to restore the previous context at.
    root_context_var.set(internal_ctx().replace_task_context(tctx))
    logger.info(f"Loaded task context for action {action.name} (run {action.run_name}) from {cfg_path}")
    return tctx

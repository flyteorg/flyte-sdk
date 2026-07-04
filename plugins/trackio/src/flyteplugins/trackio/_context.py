from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import flyte

_TRACKIO_RUN_KEY = "_trackio_run"

def _to_dict_helper(obj, prefix: str) -> dict[str, str]:
    """Serialize a dataclass into Flyte custom_context."""

    result: dict[str, str] = {}

    for key, value in asdict(obj).items():
        if value is None:
            continue

        if isinstance(value, (dict, list, bool)):
            result[f"{prefix}_{key}"] = json.dumps(value)
        else:
            result[f"{prefix}_{key}"] = str(value)

    return result


def _from_dict_helper(cls, d: dict[str, str], prefix: str):
    """Deserialize a dataclass from Flyte custom_context."""

    kwargs = {}

    prefix = f"{prefix}_"

    for key, value in d.items():
        if not key.startswith(prefix):
            continue

        field = key[len(prefix) :]

        try:
            kwargs[field] = json.loads(value)
        except Exception:
            kwargs[field] = value

    return cls(**kwargs)


def _context_manager_enter(obj, prefix: str):
    ctx = flyte.ctx()

    saved = {}

    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith(f"{prefix}_"):
                saved[key] = ctx.custom_context[key]

    ctx_mgr = flyte.custom_context(**obj)

    ctx_mgr.__enter__()

    return saved, ctx_mgr


def _context_manager_exit(ctx_mgr, saved: dict, prefix: str, *args):
    if ctx_mgr:
        ctx_mgr.__exit__(*args)

    ctx = flyte.ctx()

    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith(f"{prefix}_"):
                del ctx.custom_context[key]

        ctx.custom_context.update(saved)


@dataclass
class _TrackioConfig:
    """
    Trackio configuration stored inside the Flyte custom_context.
    Mirrors the supported subset of ``trackio.init()``.
    """

    project: Optional[str] = None

    name: Optional[str] = None

    group: Optional[str] = None

    space_id: Optional[str] = None

    dataset_id: Optional[str] = None

    bucket_id: Optional[str] = None

    server_url: Optional[str] = None

    config: Optional[dict[str, Any]] = None

    resume: str = "never"

    auto_log_gpu: Optional[bool] = None

    gpu_log_interval: float = 10.0

    auto_log_cpu: Optional[bool] = None

    cpu_log_interval: float = 10.0

    def to_trackio_init(self) -> dict[str, Any]:
        """Convert to arguments for ``trackio.init()``."""

        return {
            k: v
            for k, v in asdict(self).items()
            if v is not None
        }

    def to_dict(self) -> dict[str, str]:
        return _to_dict_helper(self, "trackio")

    @classmethod
    def from_dict(cls, d: dict[str, str]):
        return _from_dict_helper(cls, d, "trackio")

    def __enter__(self):
        self._saved, self._ctx = _context_manager_enter(
            self,
            "trackio",
        )
        return self

    def __exit__(self, *args):
        _context_manager_exit(
            self._ctx,
            self._saved,
            "trackio",
            *args,
        )


def get_trackio_context() -> Optional[_TrackioConfig]:
    """
    Return the current Trackio configuration.
    """

    ctx = flyte.ctx()

    if ctx is None or not ctx.custom_context:
        return None

    if not any(
        k.startswith("trackio_")
        for k in ctx.custom_context
    ):
        return None

    return _TrackioConfig.from_dict(
        ctx.custom_context
    )


def trackio_config(
    *,
    project: str | None = None,
    name: str | None = None,
    group: str | None = None,
    space_id: str | None = None,
    dataset_id: str | None = None,
    bucket_id: str | None = None,
    server_url: str | None = None,
    config: dict[str, Any] | None = None,
    resume: str = "never",
    auto_log_gpu: bool | None = None,
    gpu_log_interval: float = 10.0,
    auto_log_cpu: bool | None = None,
    cpu_log_interval: float = 10.0,
) -> _TrackioConfig:
    """
    Create Trackio configuration for Flyte.
    """

    return _TrackioConfig(
        project=project,
        name=name,
        group=group,
        space_id=space_id,
        dataset_id=dataset_id,
        bucket_id=bucket_id,
        server_url=server_url,
        config=config,
        resume=resume,
        auto_log_gpu=auto_log_gpu,
        gpu_log_interval=gpu_log_interval,
        auto_log_cpu=auto_log_cpu,
        cpu_log_interval=cpu_log_interval,
    )




def set_trackio_run(run) -> None:
    """Store the active Trackio run in the Flyte context."""

    ctx = flyte.ctx()

    if ctx is None:
        return

    if ctx.data is None:
        ctx.data = {}

    ctx.data[_TRACKIO_RUN_KEY] = run


def get_trackio_run():
    """
    Return the active Trackio run.

    If called inside a ``@trackio_init`` decorated Flyte task, this returns the
    Trackio run managed by the plugin. Otherwise it falls back to Trackio's
    globally active run (if one exists).

    Returns:
        trackio.Run | None: The active Trackio run.
    """
    ctx = flyte.ctx()

    if ctx and ctx.data:
        run = ctx.data.get(_TRACKIO_RUN_KEY)
        if run is not None:
            return run

    return getattr(trackio, "run", None)


def clear_trackio_run() -> None:
    """Remove the Trackio run from the Flyte context."""

    ctx = flyte.ctx()

    if ctx and ctx.data:
        ctx.data.pop(_TRACKIO_RUN_KEY, None)
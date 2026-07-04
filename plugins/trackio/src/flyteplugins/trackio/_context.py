from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import flyte


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

        field = key[len(prefix):]

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

    Explicit fields are used by the Flyte plugin itself.

    Any additional Trackio parameters are forwarded through ``kwargs``.
    """

    project: Optional[str] = None

    name: Optional[str] = None

    tags: Optional[list[str]] = None

    config: Optional[dict[str, Any]] = None

    group: Optional[str] = None

    notes: Optional[str] = None

    space_id: Optional[str] = None

    bucket_id: Optional[str] = None

    server_url: Optional[str] = None

    private: Optional[bool] = None

    kwargs: Optional[dict[str, Any]] = None

    def to_trackio_init(self) -> dict[str, Any]:
        """
        Convert to arguments for trackio.init().
        """

        cfg = asdict(self)

        extra = cfg.pop("kwargs", None)

        if extra:
            cfg.update(extra)

        return {
            k: v
            for k, v in cfg.items()
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
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    group: str | None = None,
    notes: str | None = None,
    space_id: str | None = None,
    bucket_id: str | None = None,
    server_url: str | None = None,
    private: bool | None = None,
    **kwargs,
) -> _TrackioConfig:
    """
    Create Trackio configuration for Flyte.
    """

    return _TrackioConfig(
        project=project,
        name=name,
        tags=tags,
        config=config,
        group=group,
        notes=notes,
        space_id=space_id,
        bucket_id=bucket_id,
        server_url=server_url,
        private=private,
        kwargs=kwargs or None,
    )


def merge_trackio_config(
    context: _TrackioConfig | None,
    decorator_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge context configuration with decorator arguments.

    Decorator arguments always win.
    """

    if context is None:
        return decorator_kwargs.copy()

    merged = context.to_trackio_init()

    merged.update(
        {
            k: v
            for k, v in decorator_kwargs.items()
            if v is not None
        }
    )

    return merged
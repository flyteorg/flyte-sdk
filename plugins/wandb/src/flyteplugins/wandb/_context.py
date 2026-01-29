import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import flyte

RunMode = Literal["auto", "new", "shared"]


def _to_dict_helper(obj, prefix: str) -> dict[str, str]:
    """Convert dataclass to string dict for Flyte's custom_context."""
    result = {}
    for key, value in asdict(obj).items():
        if value is not None:
            if isinstance(value, (list, dict, bool)):
                # Use json.dumps for lists, dicts, and bools for proper serialization
                try:
                    result[f"{prefix}_{key}"] = json.dumps(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"wandb config field '{key}' must be JSON-serializable. "
                        f"Got type: {type(value).__name__}. Error: {e}"
                    ) from e
            else:
                result[f"{prefix}_{key}"] = str(value)
    return result


def _from_dict_helper(cls, d: dict[str, str], prefix: str):
    """Create dataclass from custom_context dict."""
    kwargs = {}
    prefix_with_underscore = f"{prefix}_"
    prefix_len = len(prefix_with_underscore)

    # Exclude keys that match longer/more-specific prefixes
    # (e.g., when processing "wandb", exclude "wandb_sweep")
    exclude_prefixes = []
    if prefix == "wandb":
        exclude_prefixes = ["wandb_sweep_"]

    for key, value in d.items():
        if key.startswith(prefix_with_underscore):
            # Skip if this key matches a more specific prefix
            if any(key.startswith(excl) for excl in exclude_prefixes):
                continue

            field_name = key[prefix_len:]
            try:
                kwargs[field_name] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                kwargs[field_name] = value
    return cls(**kwargs)


def _context_manager_enter(obj, prefix: str):
    """Generic __enter__ for wandb config context managers."""
    ctx = flyte.ctx()
    saved_config = {}
    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith(f"{prefix}_"):
                saved_config[key] = ctx.custom_context[key]

    ctx_mgr = flyte.custom_context(**obj)
    ctx_mgr.__enter__()
    return saved_config, ctx_mgr


def _context_manager_exit(ctx_mgr, saved_config: dict, prefix: str, *args):
    """Generic __exit__ for wandb config context managers."""
    if ctx_mgr:
        ctx_mgr.__exit__(*args)

    ctx = flyte.ctx()
    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith(f"{prefix}_"):
                del ctx.custom_context[key]
        ctx.custom_context.update(saved_config)


@dataclass
class _WandBConfig:
    """
    Pass any other wandb.init() parameters via kwargs dict:
      - notes, job_type, save_code
      - resume, resume_from, fork_from, reinit
      - anonymous, allow_val_change, force
      - settings, and more

    See: https://docs.wandb.ai/ref/python/init
    """

    # Essential fields (most commonly used)
    project: Optional[str] = None
    entity: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[list[str]] = None
    config: Optional[dict[str, Any]] = None

    # Common optional fields
    mode: Optional[str] = None
    group: Optional[str] = None

    # Flyte-specific run mode (not passed to wandb.init)
    # Controls whether to create a new W&B run or share an existing one
    run_mode: RunMode = "auto"  # "auto", "new", or "shared"

    # Flyte-specific: download wandb logs after task completes
    download_logs: bool = False

    # Catch-all for additional wandb.init() parameters
    kwargs: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, str]:
        """Convert to string dict for Flyte's custom_context."""
        return _to_dict_helper(self, "wandb")

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "_WandBConfig":
        """Create from custom_context dict."""
        return _from_dict_helper(cls, d, "wandb")

    # Dict protocol - for ** unpacking
    def keys(self):
        return self.to_dict().keys()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        # For setting items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            ctx.custom_context[key] = value

    def __delitem__(self, key):
        # For deleting items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            del ctx.custom_context[key]

    def items(self):
        return self.to_dict().items()

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def pop(self, key, default=None):
        # For popping items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            return ctx.custom_context.pop(key, default)
        return default

    def update(self, *args, **kwargs):
        # For updating, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            ctx.custom_context.update(*args, **kwargs)

    # Context manager implementation
    def __enter__(self):
        self._saved_config, self._ctx = _context_manager_enter(self, "wandb")
        return self

    def __exit__(self, *args):
        _context_manager_exit(self._ctx, self._saved_config, "wandb", *args)


def get_wandb_context() -> Optional[_WandBConfig]:
    """Get wandb config from current Flyte context."""
    ctx = flyte.ctx()
    if ctx is None or not ctx.custom_context:
        return None

    # Check if we have wandb_ prefixed keys
    has_wandb_keys = any(k.startswith("wandb_") for k in ctx.custom_context.keys())
    if not has_wandb_keys:
        return None

    return _WandBConfig.from_dict(ctx.custom_context)


def wandb_config(
    project: Optional[str] = None,
    entity: Optional[str] = None,
    id: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    config: Optional[dict[str, Any]] = None,
    mode: Optional[str] = None,
    group: Optional[str] = None,
    run_mode: RunMode = "auto",
    download_logs: bool = False,
    **kwargs: Any,
) -> _WandBConfig:
    """
    Create wandb configuration.

    This function works in two contexts:
    1. With `flyte.with_runcontext()` - sets global wandb config
    2. As a context manager - overrides config for specific tasks

    Args:
        project: W&B project name
        entity: W&B entity (team or username)
        id: Unique run id (auto-generated if not provided)
        name: Human-readable run name
        tags: List of tags for organizing runs
        config: Dictionary of hyperparameters
        mode: "online", "offline" or "disabled"
        group: Group name for related runs
        run_mode: Flyte-specific run mode - "auto", "new" or "shared".
            Controls whether tasks create new W&B runs or share existing ones
        download_logs: If `True`, downloads wandb run files after task completes
            and shows them as a trace output in the Flyte UI
        **kwargs: Additional `wandb.init()` parameters
    """
    return _WandBConfig(
        project=project,
        entity=entity,
        id=id,
        name=name,
        tags=tags,
        config=config,
        mode=mode,
        group=group,
        run_mode=run_mode,
        download_logs=download_logs,
        kwargs=kwargs if kwargs else None,
    )


@dataclass
class _WandBSweepConfig:
    # Essential sweep parameters
    name: Optional[str] = None
    method: Optional[str] = None
    metric: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None

    # Sweep metadata
    project: Optional[str] = None
    entity: Optional[str] = None
    prior_runs: Optional[list[str]] = None

    # Flyte-specific: download wandb sweep logs after task completes
    download_logs: bool = False

    # Catch-all for additional sweep config parameters
    # (e.g. early_terminate, name, description, command, controller, etc.)
    kwargs: Optional[dict[str, Any]] = None

    def to_sweep_config(self) -> dict[str, Any]:
        """Convert to wandb.sweep() compatible dict."""
        config = asdict(self)

        # Remove fields that aren't part of the sweep config
        config.pop("project", None)
        config.pop("entity", None)
        config.pop("prior_runs", None)
        config.pop("download_logs", None)

        # Merge kwargs into the main config
        extra_kwargs = config.pop("kwargs", None)
        if extra_kwargs:
            config.update(extra_kwargs)

        # Remove None values
        return {k: v for k, v in config.items() if v is not None}

    def to_dict(self) -> dict[str, str]:
        """Convert to string dict for Flyte's custom_context."""
        return _to_dict_helper(self, "wandb_sweep")

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "_WandBSweepConfig":
        """Create from custom_context dict."""
        return _from_dict_helper(cls, d, "wandb_sweep")

    # Dict protocol - for ** unpacking
    def keys(self):
        return self.to_dict().keys()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        # For setting items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            ctx.custom_context[key] = value

    def __delitem__(self, key):
        # For deleting items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            del ctx.custom_context[key]

    def items(self):
        return self.to_dict().items()

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def pop(self, key, default=None):
        # For popping items, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            return ctx.custom_context.pop(key, default)
        return default

    def update(self, *args, **kwargs):
        # For updating, we need to update the actual Flyte context
        ctx = flyte.ctx()
        if ctx and ctx.custom_context:
            ctx.custom_context.update(*args, **kwargs)

    # Context manager implementation
    def __enter__(self):
        self._saved_config, self._ctx = _context_manager_enter(self, "wandb_sweep")
        return self

    def __exit__(self, *args):
        _context_manager_exit(self._ctx, self._saved_config, "wandb_sweep", *args)


def get_wandb_sweep_context() -> Optional[_WandBSweepConfig]:
    """Get wandb sweep config from current Flyte context."""
    ctx = flyte.ctx()
    if ctx is None or not ctx.custom_context:
        return None

    has_wandb_sweep_keys = any(k.startswith("wandb_sweep_") for k in ctx.custom_context.keys())
    if not has_wandb_sweep_keys:
        return None

    return _WandBSweepConfig.from_dict(ctx.custom_context)


def wandb_sweep_config(
    method: Optional[str] = None,
    metric: Optional[dict[str, Any]] = None,
    parameters: Optional[dict[str, Any]] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    prior_runs: Optional[list[str]] = None,
    name: Optional[str] = None,
    download_logs: bool = False,
    **kwargs: Any,
) -> _WandBSweepConfig:
    """
    Create wandb sweep configuration for hyperparameter optimization.

    Args:
        method: Sweep method (e.g., "random", "grid", "bayes")
        metric: Metric to optimize (e.g., {"name": "loss", "goal": "minimize"})
        parameters: Parameter definitions for the sweep
        project: W&B project for the sweep
        entity: W&B entity for the sweep
        prior_runs: List of prior run IDs to include in the sweep analysis
        name: Sweep name (auto-generated as `{run_name}-{action_name}` if not provided)
        download_logs: If `True`, downloads all sweep run files after task completes
            and shows them as a trace output in the Flyte UI
        **kwargs: additional sweep config parameters like `early_terminate`, `description`, `command`, etc.

    See: https://docs.wandb.ai/models/sweeps/sweep-config-keys
    """
    return _WandBSweepConfig(
        name=name,
        method=method,
        metric=metric,
        parameters=parameters,
        project=project,
        entity=entity,
        prior_runs=prior_runs,
        download_logs=download_logs,
        kwargs=kwargs if kwargs else None,
    )

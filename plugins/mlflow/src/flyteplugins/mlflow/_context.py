import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import flyte

RunMode = Literal["auto", "new", "nested"]


def _to_dict_helper(obj) -> dict[str, str]:
    """
    Convert config dataclass to Flyte custom_context format.
    All keys are stored as mlflow_* strings.
    """
    result: dict[str, str] = {}

    for key, value in asdict(obj).items():
        if value is None:
            continue

        if isinstance(value, (list, dict, bool)):
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"mlflow config field '{key}' must be JSON-serializable. "
                    f"Got type: {type(value).__name__}. Error: {e}"
                ) from e
        else:
            serialized = str(value)

        result[f"mlflow_{key}"] = serialized

    return result


def _from_dict_helper(cls, d: dict[str, str]):
    """
    Convert Flyte custom_context dictionary back to config dataclass.
    """
    kwargs = {}

    for key, value in d.items():
        if key.startswith("mlflow_"):
            field = key[len("mlflow_") :]

            try:
                kwargs[field] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                kwargs[field] = value

    return cls(**kwargs)


def _context_manager_enter(obj):
    ctx = flyte.ctx()
    saved_config = {}

    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith("mlflow_"):
                saved_config[key] = ctx.custom_context[key]

    ctx_mgr = flyte.custom_context(**obj.to_dict())
    ctx_mgr.__enter__()

    return saved_config, ctx_mgr


def _context_manager_exit(ctx_mgr, saved_config: dict, *args):
    if ctx_mgr:
        ctx_mgr.__exit__(*args)

    ctx = flyte.ctx()

    if ctx and ctx.custom_context:
        for key in list(ctx.custom_context.keys()):
            if key.startswith("mlflow_"):
                del ctx.custom_context[key]

        ctx.custom_context.update(saved_config)


@dataclass
class _MLflowConfig:
    """
    MLflow configuration.

    Handles both manual logging and autologging.
    """

    # Tracking configuration
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None

    # Run configuration
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    tags: Optional[dict[str, str]] = None

    # Flyte-specific run mode
    run_mode: RunMode = "auto"

    # Autolog configuration
    autolog: bool = False
    framework: Optional[str] = None
    log_models: Optional[bool] = None
    log_datasets: Optional[bool] = None
    autolog_kwargs: Optional[dict[str, Any]] = None

    # Link configuration
    link_host: Optional[str] = None
    link_template: Optional[str] = None

    # Extra mlflow.start_run kwargs
    kwargs: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, str]:
        return _to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, str]):
        return _from_dict_helper(cls, d)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def __enter__(self):
        self._saved_config, self._ctx = _context_manager_enter(self)
        return self

    def __exit__(self, *args):
        _context_manager_exit(self._ctx, self._saved_config, *args)


def get_mlflow_context() -> Optional[_MLflowConfig]:
    """
    Retrieve current MLflow configuration from Flyte context.
    """
    ctx = flyte.ctx()

    if ctx is None or not ctx.custom_context:
        return None

    has_mlflow_keys = any(k.startswith("mlflow_") for k in ctx.custom_context)

    if not has_mlflow_keys:
        return None

    return _MLflowConfig.from_dict(ctx.custom_context)


def mlflow_config(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    run_name: Optional[str] = None,
    run_id: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    run_mode: RunMode = "auto",
    autolog: bool = False,
    framework: Optional[str] = None,
    log_models: Optional[bool] = None,
    log_datasets: Optional[bool] = None,
    autolog_kwargs: Optional[dict[str, Any]] = None,
    link_host: Optional[str] = None,
    link_template: Optional[str] = None,
    **kwargs: Any,
) -> _MLflowConfig:
    """
    Create MLflow configuration.

    Works in two contexts:
    1. With `flyte.with_runcontext()` for global configuration
    2. As a context manager to override configuration

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name.
        experiment_id: MLflow experiment ID.
        run_name: Human-readable run name.
        run_id: Explicit MLflow run ID.
        tags: MLflow run tags.
        run_mode: Flyte-specific run mode ("auto", "new", "nested").
        autolog: Enable MLflow autologging.
        framework: Framework-specific autolog (e.g. "sklearn", "pytorch").
        log_models: Whether to log models automatically.
        log_datasets: Whether to log datasets automatically.
        autolog_kwargs: Extra parameters passed to mlflow.autolog().
        link_host: MLflow UI host for auto-generating task links.
        link_template: Custom URL template. Defaults to standard MLflow UI format.
            Available placeholders: `{host}`, `{experiment_id}`, `{run_id}`.
        **kwargs: Extra parameters passed to mlflow.start_run().
    """

    if experiment_name and experiment_id:
        raise ValueError("Cannot provide both 'experiment_name' and 'experiment_id'.")

    if run_name and run_id:
        raise ValueError("Cannot provide both 'run_name' and 'run_id'.")

    return _MLflowConfig(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
        run_name=run_name,
        run_id=run_id,
        tags=tags,
        run_mode=run_mode,
        autolog=autolog,
        framework=framework,
        log_models=log_models,
        log_datasets=log_datasets,
        autolog_kwargs=autolog_kwargs,
        link_host=link_host,
        link_template=link_template,
        kwargs=kwargs or None,
    )

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import flyte


@dataclass
class _WandBConfig:
    """
    Additional parameters via kwargs:
        Pass any other wandb.init() parameters via kwargs dict:
        - notes, dir, job_type, save_code
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

    # Catch-all for additional wandb.init() parameters
    kwargs: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, str]:
        """Convert to string dict for Flyte's custom_context."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    # Serialize complex types as JSON
                    try:
                        result[f"wandb_{key}"] = json.dumps(value)
                    except (TypeError, ValueError) as e:
                        raise ValueError(
                            f"wandb config field '{key}' must be JSON-serializable. "
                            f"Got type: {type(value).__name__}. Error: {e}"
                        ) from e
                else:
                    result[f"wandb_{key}"] = str(value)
        return result

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "_WandBConfig":
        """Create from custom_context dict."""
        kwargs = {}
        for key, value in d.items():
            if key.startswith("wandb_"):
                field_name = key[6:]  # Remove "wandb_" prefix
                # Try to parse JSON for lists/dicts/bools
                try:
                    kwargs[field_name] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    kwargs[field_name] = value
        return cls(**kwargs)

    # Dict protocol - minimal implementation for ** unpacking
    def keys(self):
        return self.to_dict().keys()

    def __getitem__(self, key):
        return self.to_dict()[key]

    # Context manager implementation
    def __enter__(self):
        self._ctx = flyte.custom_context(**self).__enter__()
        return self

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)


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
    **kwargs: Any,
) -> _WandBConfig:
    """
    Create wandb configuration.

    This function works in two contexts:
    1. With flyte.with_runcontext() - sets global wandb config
    2. As a context manager - overrides config for specific tasks

    Args:
        project: wandb project name
        entity: wandb entity (team or username)
        id: unique run id (auto-generated if not provided)
        name: human-readable run name
        tags: list of tags for organizing runs
        config: dictionary of hyperparameters
        mode: "online", "offline", or "disabled"
        group: group name for related runs
        **kwargs: additional wandb.init() parameters
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
        kwargs=kwargs if kwargs else None,
    )

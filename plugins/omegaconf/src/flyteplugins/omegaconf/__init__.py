"""OmegaConf DictConfig/ListConfig support for Flyte."""

from __future__ import annotations

import functools

from flyte.types._type_engine import TypeEngine

from .base_transformer import DictConfigTransformer, ListConfigTransformer


@functools.lru_cache(maxsize=None)
def register_omegaconf_transformers() -> None:
    """Register OmegaConf transformers with Flyte TypeEngine.

    Called via the ``flyte.plugins.types`` entry point on import, or manually
    by importing this package.
    """
    TypeEngine.register(DictConfigTransformer())
    TypeEngine.register(ListConfigTransformer())


# Register at module import time for backwards compatibility.
register_omegaconf_transformers()

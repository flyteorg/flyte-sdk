"""OmegaConf DictConfig/ListConfig support for Flyte.

Importing this package registers the transformers with Flyte's TypeEngine,
enabling DictConfig and ListConfig as typed task inputs and outputs.
"""

from __future__ import annotations

import functools

from flyte.types._type_engine import TypeEngine

from .dictconfig_transformer import DictConfigTransformer
from .listconfig_transformer import ListConfigTransformer


@functools.lru_cache(maxsize=None)
def register_omegaconf_transformers() -> None:
    """Register OmegaConf transformers with Flyte TypeEngine.

    Called via the ``flyte.plugins.types`` entry point on import, or manually
    by importing this package.
    """
    TypeEngine.register(DictConfigTransformer())
    TypeEngine.register(ListConfigTransformer())


# Register on import so that `import flyteplugins.omegaconf` is sufficient.
register_omegaconf_transformers()

"""
# Flyte Type System

The Flyte type system provides a way to define, transform, and manipulate types in Flyte workflows.
Since the data flowing through Flyte has to often cross process, container and langauge boundaries, the type system
is designed to be serializable to a universal format that can be understood across different environments. This
universal format is based on Protocol Buffers. The types are called LiteralTypes and the runtime
representation of data is called Literals.

The type system includes:
- **TypeEngine**: The core engine that manages type transformations and serialization. This is the main entry point for
  for all the internal type transformations and serialization logic.
- **TypeTransformer**: A class that defines how to transform one type to another. This is extensible
    allowing users to define custom types and transformations.
- **Renderable**: An interface for types that can be rendered as HTML, that can be outputted to a flyte.report.

It is always possible to bypass the type system and use the `FlytePickle` type to serialize any python object
 into a pickle format. The pickle format is not human-readable, but can be passed between flyte tasks that are
 written in python. The Pickled objects cannot be represented in the UI, and may be in-efficient for large datasets.
"""

from importlib.metadata import entry_points

from flyte._logging import logger

from ._interface import guess_interface
from ._pickle import FlytePickle
from ._renderer import Renderable
from ._string_literals import literal_string_repr
from ._type_engine import TypeEngine, TypeTransformer, TypeTransformerFailedError

__all__ = [
    "FlytePickle",
    "Renderable",
    "TypeEngine",
    "TypeTransformer",
    "TypeTransformerFailedError",
    "guess_interface",
    "literal_string_repr",
]


def _load_custom_type_transformers():
    plugins = entry_points(group="flyte.plugins.types")
    for ep in plugins:
        try:
            logger.info(f"Loading type transformer: {ep.name}")
            loaded = ep.load()
            if callable(loaded):
                loaded()
        except Exception as e:
            logger.warning(f"Failed to load type transformer {ep.name} with error: {e}")

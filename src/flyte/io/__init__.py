"""
## IO data types

This package contains additional data types beyond the primitive data types in python to abstract data flow
of large datasets in Union.

"""

__all__ = [
    "PARQUET",
    "DataFrame",
    "Dir",
    "EmptyDir",
    "File",
    "HashFunction",
]

from typing import TYPE_CHECKING, Any

from ._dir import Dir, EmptyDir
from ._file import File
from ._hashing_io import HashFunction

if TYPE_CHECKING:
    from ._dataframe import PARQUET, DataFrame


def __getattr__(name: str) -> Any:
    # DataFrame pulls in the structured-dataset engine (~0.2s); File and Dir do
    # not need it, so it loads on first use (PEP 562). Importing it registers
    # its transformer, as before.
    if name in ("DataFrame", "PARQUET"):
        from . import _dataframe

        value = getattr(_dataframe, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

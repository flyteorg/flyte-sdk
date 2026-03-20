import importlib
import sys

from .transformers.pandas import register_pandera_pandas_type_transformers
from .transformers.polars import register_pandera_polars_type_transformers


def _check_package(package_name: str) -> bool:
    return package_name in sys.modules or importlib.import_module(package_name) is not None


def register_type_transformers() -> None:
    """Register all flyteplugins-pandera type transformers."""
    if _check_package("pandas"):
        register_pandera_pandas_type_transformers()
    if _check_package("polars"):
        register_pandera_polars_type_transformers()

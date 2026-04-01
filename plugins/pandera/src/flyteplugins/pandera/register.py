import importlib
import sys


def _check_package(package_name: str) -> bool:
    if package_name in sys.modules:
        return True
    try:
        importlib.import_module(package_name)
        return True
    except (ModuleNotFoundError, ImportError):
        return False


def register_type_transformers() -> None:
    """Register all flyteplugins-pandera type transformers."""
    if _check_package("pandas"):
        from .transformers.pandas import register_pandera_pandas_type_transformers

        register_pandera_pandas_type_transformers()
    if _check_package("polars"):
        from .transformers.polars import register_pandera_polars_type_transformers

        register_pandera_polars_type_transformers()
    if _check_package("pyspark"):
        from .transformers.pyspark_sql import register_pandera_pyspark_sql_type_transformers

        register_pandera_pyspark_sql_type_transformers()

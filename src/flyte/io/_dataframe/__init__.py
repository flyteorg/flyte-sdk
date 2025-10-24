"""
Flytekit DataFrame
==========================================================
.. currentmodule:: flyte.io._dataframe

.. autosummary::
   :template: custom.rst
   :toctree: generated/

    DataFrame
    DataFrameDecoder
    DataFrameEncoder
"""

import functools

from flyte._logging import logger
from flyte._utils.lazy_module import is_imported

from .dataframe import (
    PARQUET,
    DataFrame,
    DataFrameDecoder,
    DataFrameEncoder,
    DataFrameTransformerEngine,
    DuplicateHandlerError,
)


@functools.lru_cache(maxsize=None)
def register_csv_handlers():
    from .basic_dfs import CSVToPandasDecodingHandler, PandasToCSVEncodingHandler

    DataFrameTransformerEngine.register(PandasToCSVEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(CSVToPandasDecodingHandler(), default_format_for_type=True)


@functools.lru_cache(maxsize=None)
def register_pandas_handlers():
    import pandas as pd

    from flyte.types._renderer import TopFrameRenderer

    from .basic_dfs import PandasToParquetEncodingHandler, ParquetToPandasDecodingHandler

    DataFrameTransformerEngine.register(PandasToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToPandasDecodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register_renderer(pd.DataFrame, TopFrameRenderer())


@functools.lru_cache(maxsize=None)
def register_arrow_handlers():
    import pyarrow as pa

    from flyte.types._renderer import ArrowRenderer

    from .basic_dfs import ArrowToParquetEncodingHandler, ParquetToArrowDecodingHandler

    DataFrameTransformerEngine.register(ArrowToParquetEncodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register(ParquetToArrowDecodingHandler(), default_format_for_type=True)
    DataFrameTransformerEngine.register_renderer(pa.Table, ArrowRenderer())


@functools.lru_cache(maxsize=None)
def register_bigquery_handlers():
    try:
        from .bigquery import (
            ArrowToBQEncodingHandlers,
            BQToArrowDecodingHandler,
            BQToPandasDecodingHandler,
            PandasToBQEncodingHandlers,
        )

        DataFrameTransformerEngine.register(PandasToBQEncodingHandlers())
        DataFrameTransformerEngine.register(BQToPandasDecodingHandler())
        DataFrameTransformerEngine.register(ArrowToBQEncodingHandlers())
        DataFrameTransformerEngine.register(BQToArrowDecodingHandler())
    except ImportError:
        logger.info(
            "We won't register bigquery handler for structured dataset because "
            "we can't find the packages google-cloud-bigquery-storage and google-cloud-bigquery"
        )


@functools.lru_cache(maxsize=None)
def register_snowflake_handlers():
    try:
        from .snowflake import PandasToSnowflakeEncodingHandlers, SnowflakeToPandasDecodingHandler

        DataFrameTransformerEngine.register(SnowflakeToPandasDecodingHandler())
        DataFrameTransformerEngine.register(PandasToSnowflakeEncodingHandlers())

    except ImportError:
        logger.info(
            "We won't register snowflake handler for structured dataset because "
            "we can't find package snowflake-connector-python"
        )


def lazy_import_dataframe_handler():
    if is_imported("pandas"):
        try:
            register_pandas_handlers()
            register_csv_handlers()
        except DuplicateHandlerError:
            logger.debug("Transformer for pandas is already registered.")
    if is_imported("pyarrow"):
        try:
            register_arrow_handlers()
        except DuplicateHandlerError:
            logger.debug("Transformer for arrow is already registered.")
    if is_imported("google.cloud.bigquery"):
        try:
            register_bigquery_handlers()
        except DuplicateHandlerError:
            logger.debug("Transformer for bigquery is already registered.")
    if is_imported("snowflake.connector"):
        try:
            register_snowflake_handlers()
        except DuplicateHandlerError:
            logger.debug("Transformer for snowflake is already registered.")


__all__ = [
    "PARQUET",
    "DataFrame",
    "DataFrameDecoder",
    "DataFrameEncoder",
    "DataFrameTransformerEngine",
    "lazy_import_dataframe_handler",
]

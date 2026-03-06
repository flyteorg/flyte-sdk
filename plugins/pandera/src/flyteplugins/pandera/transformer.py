"""Pandera type transformer for Flyte v2 SDK.

This module provides a TypeTransformer that validates pandas DataFrames against
Pandera schemas during serialization and deserialization in Flyte tasks.
"""

import functools
import typing
from typing import Type, Union

from flyte._logging import logger
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrameTransformerEngine
from flyte.types import TypeEngine, TypeTransformer
from flyteidl2.core import literals_pb2, types_pb2

from .config import ValidationConfig
from .renderer import PandasReportRenderer

if typing.TYPE_CHECKING:
    import pandas

    import pandera
else:
    pandas = lazy_module("pandas")
    pandera = lazy_module("pandera")

T = typing.TypeVar("T")


class PanderaTransformer(TypeTransformer["pandera.typing.DataFrame"]):
    """Type transformer for pandera.typing.DataFrame.

    Wraps the DataFrameTransformerEngine to add automatic Pandera schema
    validation when data flows between Flyte tasks.
    """

    _VALIDATION_MEMO: typing.ClassVar[set] = set()

    def __init__(self):
        super().__init__("Pandera Transformer", pandera.typing.DataFrame)
        self._sd_transformer = DataFrameTransformerEngine()

    @staticmethod
    def _get_pandera_schema(
        t: Type["pandera.typing.DataFrame"],
    ) -> tuple["pandera.DataFrameSchema", ValidationConfig]:
        config = ValidationConfig()
        if typing.get_origin(t) is typing.Annotated:
            t, *args = typing.get_args(t)
            for arg in args:
                if isinstance(arg, ValidationConfig):
                    config = arg
                    break

        type_args = typing.get_args(t)
        if type_args:
            schema_model, *_ = type_args
            schema = schema_model.to_schema()
        else:
            schema = pandera.DataFrameSchema()
        return schema, config

    def assert_type(self, t: Type[T], v: T):
        if not hasattr(t, "__origin__") and not isinstance(v, (t, pandas.DataFrame)):
            raise TypeError(f"Type of Val '{v}' is not an instance of {t}")

    def get_literal_type(self, t: Type["pandera.typing.DataFrame"]) -> types_pb2.LiteralType:
        if typing.get_origin(t) is typing.Annotated:
            t, _ = typing.get_args(t)
        return self._sd_transformer.get_literal_type(t)

    def _validate_and_report(
        self,
        df: "pandas.DataFrame",
        schema: "pandera.DataFrameSchema",
        config: ValidationConfig,
    ) -> "pandas.DataFrame":
        """Validate a DataFrame against a Pandera schema and generate a report.

        Returns the validated DataFrame (which may have coerced types).
        Raises SchemaErrors if validation fails and config.on_error == "raise".
        """
        renderer = PandasReportRenderer(title=f"Pandera Report: {schema.name}")
        try:
            val = schema.validate(df, lazy=True)
        except (pandera.errors.SchemaError, pandera.errors.SchemaErrors) as exc:
            renderer.to_html(df, schema, exc)
            val = df
            if config.on_error == "raise":
                raise
            elif config.on_error == "warn":
                logger.warning(str(exc))
            else:
                raise ValueError(f"Invalid on_error value: {config.on_error}")
        else:
            renderer.to_html(val, schema)
        return val

    async def to_literal(
        self,
        python_val: Union["pandas.DataFrame", typing.Any],
        python_type: Type["pandera.typing.DataFrame"],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        if not isinstance(python_val, pandas.DataFrame):
            raise AssertionError(
                f"Only pandas DataFrame objects can be returned from a Pandera-validated task, got {type(python_val)}"
            )

        schema, config = self._get_pandera_schema(python_type)
        val = self._validate_and_report(python_val, schema, config)

        lv = await self._sd_transformer.to_literal(val, pandas.DataFrame, expected)

        # Cache the URI + schema name to skip re-validation during local execution
        # where to_literal is followed immediately by to_python_value
        if lv.scalar and lv.scalar.structured_dataset:
            self._VALIDATION_MEMO.add((lv.scalar.structured_dataset.uri, schema.name))

        return lv

    async def to_python_value(
        self,
        lv: literals_pb2.Literal,
        expected_python_type: Type["pandera.typing.DataFrame"],
    ) -> "pandera.typing.DataFrame":
        if not (lv and lv.scalar and lv.scalar.structured_dataset):
            raise AssertionError("Can only convert a literal structured dataset to a pandera schema")

        df = await self._sd_transformer.to_python_value(lv, pandas.DataFrame)
        schema, config = self._get_pandera_schema(expected_python_type)

        # Skip validation if we already validated this data in to_literal (local execution)
        if (lv.scalar.structured_dataset.uri, schema.name) in self._VALIDATION_MEMO:
            return df

        return self._validate_and_report(df, schema, config)


@functools.lru_cache(maxsize=None)
def register_pandera_transformer():
    """Register the Pandera transformer with Flyte's type engine.

    This function is called automatically via the flyte.plugins.types entry point
    when flyte.init() is called with load_plugin_type_transformers=True (the default).
    """
    TypeEngine.register(PanderaTransformer())


# Also register at module import time for backwards compatibility
register_pandera_transformer()

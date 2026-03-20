from __future__ import annotations

import functools
import typing
from typing import Any, TypeVar

import flyte.report
from flyte._logging import logger
from flyte._utils import lazy_module
from flyte.io._dataframe.dataframe import DataFrame as FlyteDataFrame
from flyte.io._dataframe.dataframe import DataFrameTransformerEngine
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError
from flyteidl2.core.types_pb2 import LiteralType

from .config import ValidationConfig
from .renderer import PanderaReportRenderer

if typing.TYPE_CHECKING:
    import pandas as pd
else:
    pd = lazy_module("pandas")

T = TypeVar("T")


def _unwrap_annotated(t: type[T]) -> tuple[type[T], tuple[Any, ...]]:
    if typing.get_origin(t) is typing.Annotated:
        base, *metadata = typing.get_args(t)
        return typing.cast(type[T], base), tuple(metadata)
    return t, ()


def _extract_config(metadata: tuple[Any, ...]) -> ValidationConfig:
    for entry in metadata:
        if isinstance(entry, ValidationConfig):
            return entry
    return ValidationConfig()


def _pandera_origin(t: Any) -> Any:
    return typing.get_origin(t) or t


def _schema_model_from_pandera_type(t: Any) -> Any | None:
    args = typing.get_args(t)
    if not args:
        return None
    return args[0]


def _resolve_native_df_type(pandera_type: Any) -> type[Any]:
    origin = _pandera_origin(pandera_type)
    module_name = getattr(origin, "__module__", "")
    class_name = getattr(origin, "__name__", "")

    if module_name.endswith("typing.pandas") and class_name == "DataFrame":
        return pd.DataFrame
    if module_name.endswith("typing.polars") and class_name == "DataFrame":
        return lazy_module("polars").DataFrame
    if module_name.endswith("typing.polars") and class_name == "LazyFrame":
        return lazy_module("polars").LazyFrame
    if module_name.endswith("typing.ibis") and class_name == "Table":
        ibis = lazy_module("ibis")
        if hasattr(ibis, "Table"):
            return ibis.Table
        return lazy_module("ibis.expr.types").Table
    if module_name.endswith("typing.pyspark_sql") and class_name == "DataFrame":
        return lazy_module("pyspark.sql").DataFrame
    if module_name.endswith("typing.pyspark") and class_name == "DataFrame":
        return lazy_module("pyspark.pandas.frame").DataFrame
    if module_name.endswith("typing.modin") and class_name == "DataFrame":
        return lazy_module("modin.pandas").DataFrame
    if module_name.endswith("typing.dask") and class_name == "DataFrame":
        return lazy_module("dask.dataframe").DataFrame
    raise TypeTransformerFailedError(f"Unsupported pandera typing type: {origin}")


def _schema_from_pandera_type(pandera_type: Any) -> Any:
    pandera_mod = lazy_module("pandera")
    schema_model = _schema_model_from_pandera_type(pandera_type)
    if schema_model is not None and hasattr(schema_model, "to_schema"):
        return schema_model.to_schema()
    return pandera_mod.DataFrameSchema()


def _safe_get_type(module_name: str, attribute: str) -> type[Any] | None:
    try:
        return getattr(lazy_module(module_name), attribute)
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None


class PanderaDataFrameTransformer(TypeTransformer[Any]):
    def __init__(self) -> None:
        super().__init__("Pandera DataFrame Transformer", lazy_module("pandera.typing.pandas").DataFrame)
        self._df_transformer = DataFrameTransformerEngine()
        self._validation_memo: set[tuple[str, str]] = set()

    def get_literal_type(self, t: type[Any]) -> LiteralType:
        base, metadata = _unwrap_annotated(t)
        raw_type = _resolve_native_df_type(base)
        passthrough_annotations = [m for m in metadata if not isinstance(m, ValidationConfig)]
        if passthrough_annotations:
            annotated_raw_type = typing.Annotated.__class_getitem__((raw_type, *passthrough_annotations))
            return TypeEngine.to_literal_type(annotated_raw_type)
        return TypeEngine.to_literal_type(raw_type)

    def assert_type(self, t: type[T], v: T) -> None:
        base, _ = _unwrap_annotated(t)
        raw_type = _resolve_native_df_type(base)
        if isinstance(v, FlyteDataFrame):
            return
        if not isinstance(v, raw_type):
            raise TypeTransformerFailedError(f"Expected value of type {raw_type}, got {type(v)}")

    async def _validate(
        self,
        data: Any,
        pandera_type: Any,
        config: ValidationConfig,
        report_title: str,
    ) -> Any:
        schema = _schema_from_pandera_type(pandera_type)
        renderer = PanderaReportRenderer(title=report_title)
        try:
            validated = schema.validate(data, lazy=True)
        except Exception as exc:
            html = renderer.to_html(data=data, schema=schema, error=exc)
            flyte.report.get_tab(report_title).replace(html)
            if config.on_error == "raise":
                raise
            if config.on_error == "warn":
                logger.warning(str(exc))
                return data
            raise ValueError(f"Unsupported ValidationConfig.on_error value: {config.on_error}")
        html = renderer.to_html(data=validated, schema=schema)
        flyte.report.get_tab(report_title).replace(html)
        return validated

    async def to_literal(self, python_val: Any, python_type: type[Any], expected: LiteralType):
        base, metadata = _unwrap_annotated(python_type)
        config = _extract_config(metadata)
        raw_type = _resolve_native_df_type(base)
        report_title = f"Pandera Report: {_schema_from_pandera_type(base).name or raw_type.__name__}"

        if isinstance(python_val, FlyteDataFrame):
            if python_val.val is not None:
                raw_df = python_val.val
            else:
                raw_df = await python_val.open(raw_type).all()
        else:
            raw_df = python_val

        validated = await self._validate(raw_df, base, config, report_title)
        lv = await self._df_transformer.to_literal(validated, raw_type, expected)
        uri = lv.scalar.structured_dataset.uri
        self._validation_memo.add((uri, report_title))
        return lv

    async def to_python_value(self, lv, expected_python_type: type[Any]) -> Any:
        base, metadata = _unwrap_annotated(expected_python_type)
        config = _extract_config(metadata)
        raw_type = _resolve_native_df_type(base)
        report_title = f"Pandera Report: {_schema_from_pandera_type(base).name or raw_type.__name__}"

        raw_df = await self._df_transformer.to_python_value(lv, raw_type)
        uri = lv.scalar.structured_dataset.uri
        if (uri, report_title) in self._validation_memo:
            return raw_df
        return await self._validate(raw_df, base, config, report_title)


@functools.lru_cache(maxsize=None)
def register_pandera_type_transformers() -> None:
    pandera_types = [
        _safe_get_type("pandera.typing.pandas", "DataFrame"),
        _safe_get_type("pandera.typing.polars", "DataFrame"),
        _safe_get_type("pandera.typing.polars", "LazyFrame"),
        _safe_get_type("pandera.typing.ibis", "Table"),
        _safe_get_type("pandera.typing.pyspark_sql", "DataFrame"),
        _safe_get_type("pandera.typing.pyspark", "DataFrame"),
        _safe_get_type("pandera.typing.modin", "DataFrame"),
        _safe_get_type("pandera.typing.dask", "DataFrame"),
    ]
    available_types = [t for t in pandera_types if t is not None]
    if not available_types:
        return
    transformer = PanderaDataFrameTransformer()
    TypeEngine.register_additional_type(transformer, available_types[0], override=True)
    for t in available_types[1:]:
        TypeEngine.register_additional_type(transformer, t, override=True)


register_pandera_type_transformers()

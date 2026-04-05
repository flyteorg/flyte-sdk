from __future__ import annotations

import typing
from typing import Any, TypeVar

import flyte
import flyte.report
from flyte._logging import logger
from flyte.extend import lazy_module
from flyte.io._dataframe.dataframe import DataFrame as FlyteDataFrame
from flyte.io._dataframe.dataframe import DataFrameTransformerEngine
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError
from flyteidl2.core.types_pb2 import LiteralType
from typing_extensions import Annotated, get_args, get_origin

from flyteplugins.pandera.config import ValidationConfig
from flyteplugins.pandera.renderers.base import PanderaReportRenderer
from pandera.errors import SchemaErrors

T = TypeVar("T")
DF = TypeVar("DF")


def _unwrap_annotated(t: type[T]) -> tuple[type[T], tuple[Any, ...]]:
    if get_origin(t) is Annotated:
        base, *metadata = get_args(t)
        return typing.cast(type[T], base), tuple(metadata)
    return t, ()


def _extract_config(metadata: tuple[Any, ...]) -> ValidationConfig:
    for entry in metadata:
        if isinstance(entry, ValidationConfig):
            return entry
    return ValidationConfig()


def _schema_model_from_pandera_type(t: Any) -> Any | None:
    args = typing.get_args(t)
    if not args:
        return None
    return args[0]


def _schema_from_pandera_type(pandera_type: Any) -> Any:
    pandera_mod = lazy_module("pandera")
    schema_model = _schema_model_from_pandera_type(pandera_type)
    if schema_model is not None and hasattr(schema_model, "to_schema"):
        return schema_model.to_schema()
    return pandera_mod.DataFrameSchema()


class PanderaDataFrameTransformer(TypeTransformer[DF]):
    """Pandera validation plus Flyte ``DataFrameTransformerEngine`` for pandas, Polars, PySpark SQL, etc."""

    _df_transformer = DataFrameTransformerEngine()
    _report_renderer: PanderaReportRenderer | None = None
    _validation_memo: set[tuple[str, str]]

    def _resolve_native_df_type(self, pandera_type: Any) -> type[DF]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_literal_type(self, t: type[Any]) -> LiteralType:
        base, metadata = _unwrap_annotated(t)
        raw_type = self._resolve_native_df_type(base)
        passthrough_annotations = [m for m in metadata if not isinstance(m, ValidationConfig)]
        if passthrough_annotations:
            annotated_raw_type = Annotated.__class_getitem__((raw_type, *passthrough_annotations))
            return TypeEngine.to_literal_type(annotated_raw_type)
        return TypeEngine.to_literal_type(raw_type)

    def assert_type(self, t: type[T], v: T) -> None:
        base, _ = _unwrap_annotated(t)
        raw_type = self._resolve_native_df_type(base)
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
    ) -> DF:
        schema = _schema_from_pandera_type(pandera_type)
        tctx = flyte.ctx()
        emit_report = tctx is None or not tctx.in_driver_literal_conversion
        try:
            validated = schema.validate(data, lazy=True)
        except SchemaErrors as exc:
            if emit_report:
                html = self._report_renderer.to_html(
                    title=report_title,
                    data=data,
                    schema=schema,
                    error=exc,
                    warn=config.on_error == "warn",
                )
                flyte.report.get_tab(report_title).replace(html)
                await flyte.report.flush.aio()
            if config.on_error == "raise":
                raise
            if config.on_error == "warn":
                logger.warning(str(exc))
                return data
            raise ValueError(f"Unsupported ValidationConfig.on_error value: {config.on_error}")
        if emit_report:
            html = self._report_renderer.to_html(title=report_title, data=validated, schema=schema)
            flyte.report.get_tab(report_title).replace(html)
            await flyte.report.flush.aio()
        return validated

    async def to_literal(self, python_val: Any, python_type: type[Any], expected: LiteralType):
        base, metadata = _unwrap_annotated(python_type)
        config = _extract_config(metadata)
        raw_type = self._resolve_native_df_type(base)
        report_title = "Pandera report: output"

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

    async def to_python_value(self, lv, expected_python_type: type[DF]) -> DF:
        base, metadata = _unwrap_annotated(expected_python_type)
        config = _extract_config(metadata)
        raw_type = self._resolve_native_df_type(base)
        report_title = "Pandera report: input"

        raw_df = await self._df_transformer.to_python_value(lv, raw_type)
        uri = lv.scalar.structured_dataset.uri
        if (uri, report_title) in self._validation_memo:
            return raw_df
        return await self._validate(raw_df, base, config, report_title)

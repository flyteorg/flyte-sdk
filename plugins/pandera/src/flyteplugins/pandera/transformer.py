from __future__ import annotations

import functools
import importlib
import sys
import typing
from typing import Any, TypeVar

import flyte.report
from flyte._context import internal_ctx
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


@functools.cache
def _pandas_typing_dataframe_class() -> type[Any]:
    """Canonical ``pandera.typing.pandas.DataFrame`` — use ``importlib``, not ``lazy_module`` (stable identity)."""
    mod = importlib.import_module("pandera.typing.pandas")
    return mod.DataFrame


def _all_pandas_typing_dataframe_classes() -> list[type[Any]]:
    """Every distinct ``DataFrame`` class object visible after optional duplicate module loads."""
    out: list[type[Any]] = []
    seen: set[int] = set()

    def _add(cls: type[Any] | None) -> None:
        if cls is None:
            return
        i = id(cls)
        if i in seen:
            return
        seen.add(i)
        out.append(cls)

    try:
        mod = importlib.import_module("pandera.typing.pandas")
        _add(getattr(mod, "DataFrame", None))
    except (ImportError, AttributeError):
        pass
    cached = sys.modules.get("pandera.typing.pandas")
    if cached is not None:
        _add(getattr(cached, "DataFrame", None))
    try:
        import pandera.typing as ptyping

        _add(getattr(ptyping, "DataFrame", None))
    except (ImportError, AttributeError):
        pass
    return out


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


def _schema_model_from_pandera_type(t: Any) -> Any | None:
    args = typing.get_args(t)
    if not args:
        return None
    return args[0]


def _resolve_native_df_type(pandera_type: Any) -> type[Any]:
    origin = typing.get_origin(pandera_type) or pandera_type
    mod = getattr(origin, "__module__", "")
    name = getattr(origin, "__name__", "")
    if mod.endswith("typing.pandas") and name == "DataFrame":
        return pd.DataFrame
    raise TypeTransformerFailedError(f"Only pandera.typing.pandas.DataFrame is supported (got {origin!r}).")


def _schema_from_pandera_type(pandera_type: Any) -> Any:
    pandera_mod = lazy_module("pandera")
    schema_model = _schema_model_from_pandera_type(pandera_type)
    if schema_model is not None and hasattr(schema_model, "to_schema"):
        return schema_model.to_schema()
    return pandera_mod.DataFrameSchema()


class PanderaDataFrameTransformer(TypeTransformer[Any]):
    """Pandera validation + Flyte ``DataFrameTransformerEngine`` for ``pandera.typing.pandas.DataFrame`` only."""

    def __init__(self) -> None:
        super().__init__("Pandera DataFrame Transformer", _pandas_typing_dataframe_class())
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
        emit_report = not internal_ctx().data.type_transformer_quiet
        try:
            validated = schema.validate(data, lazy=True)
        except Exception as exc:
            if emit_report:
                html = renderer.to_html(data=data, schema=schema, error=exc)
                flyte.report.get_tab(report_title).replace(html)
                await flyte.report.flush.aio()
            if config.on_error == "raise":
                raise
            if config.on_error == "warn":
                logger.warning(str(exc))
                return data
            raise ValueError(f"Unsupported ValidationConfig.on_error value: {config.on_error}")
        if emit_report:
            html = renderer.to_html(data=validated, schema=schema)
            flyte.report.get_tab(report_title).replace(html)
            await flyte.report.flush.aio()
        return validated

    async def to_literal(self, python_val: Any, python_type: type[Any], expected: LiteralType):
        base, metadata = _unwrap_annotated(python_type)
        config = _extract_config(metadata)
        raw_type = _resolve_native_df_type(base)
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

    async def to_python_value(self, lv, expected_python_type: type[Any]) -> Any:
        base, metadata = _unwrap_annotated(expected_python_type)
        config = _extract_config(metadata)
        raw_type = _resolve_native_df_type(base)
        report_title = "Pandera report: input"

        raw_df = await self._df_transformer.to_python_value(lv, raw_type)
        uri = lv.scalar.structured_dataset.uri
        if (uri, report_title) in self._validation_memo:
            return raw_df
        return await self._validate(raw_df, base, config, report_title)


def register_pandera_type_transformers() -> None:
    """Register one transformer instance for every distinct ``pandera.typing.pandas.DataFrame`` class object."""
    classes = _all_pandas_typing_dataframe_classes()
    # A plain ``import pandera.typing.pandas`` after heavy Flyte imports can expose another class object;
    # collect again so entry-point registration matches user annotations.
    try:
        import pandera.typing.pandas as _pt  # noqa: F401
    except ImportError:
        pass
    classes = list(dict.fromkeys([*classes, *_all_pandas_typing_dataframe_classes()]))

    if not classes:
        logger.warning("flyteplugins-pandera: could not resolve pandera.typing.pandas.DataFrame; is pandera installed?")
        return

    transformer = PanderaDataFrameTransformer()
    for cls in classes:
        TypeEngine.register_additional_type(transformer, cls, override=True)
        logger.debug("Registered PanderaDataFrameTransformer for %r", cls)

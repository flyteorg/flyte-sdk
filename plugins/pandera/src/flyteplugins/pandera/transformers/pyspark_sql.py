from __future__ import annotations

import importlib
import sys
import typing
from typing import Any

import flyte
import flyte.report
from flyte._logging import logger
from flyte.extend import lazy_module
from flyte.types import TypeEngine, TypeTransformerFailedError

from flyteplugins.pandera.config import ValidationConfig
from flyteplugins.pandera.renderers.pyspark_sql import PanderaPySparkSqlReportRenderer
from flyteplugins.pandera.transformers.base import _schema_from_pandera_type

from .base import PanderaDataFrameTransformer

if typing.TYPE_CHECKING:
    import pyspark.sql as psql

    import pandera.typing.pyspark_sql as pt
else:
    psql = lazy_module("pyspark.sql")
    pt = lazy_module("pandera.typing.pyspark_sql")


class PanderaPySparkSqlDataFrameTransformer(PanderaDataFrameTransformer[pt.DataFrame]):
    _report_renderer: PanderaPySparkSqlReportRenderer = PanderaPySparkSqlReportRenderer()

    def __init__(self) -> None:
        super().__init__("Pandera PySpark SQL Transformer", pt.DataFrame)
        self._validation_memo: set[tuple[str, str]] = set()

    def _resolve_native_df_type(self, pandera_type: Any) -> type[Any]:
        origin = typing.get_origin(pandera_type) or pandera_type
        mod = getattr(origin, "__module__", "")
        name = getattr(origin, "__name__", "")
        if mod.endswith("typing.pyspark_sql") and name == "DataFrame":
            return psql.DataFrame
        raise TypeTransformerFailedError(f"Only pandera.typing.pyspark_sql.DataFrame is supported (got {origin!r}).")

    async def _validate(
        self,
        data: Any,
        pandera_type: Any,
        config: ValidationConfig,
        report_title: str,
    ) -> psql.DataFrame:
        schema = _schema_from_pandera_type(pandera_type)
        tctx = flyte.ctx()
        emit_report = tctx is None or not tctx.in_driver_literal_conversion
        validated = schema.validate(data, lazy=True)
        errors = validated.pandera.errors
        if errors:
            if emit_report:
                html = self._report_renderer.to_html(
                    title=report_title,
                    data=data,
                    schema=schema,
                    error=errors,
                    warn=config.on_error == "warn",
                )
                flyte.report.get_tab(report_title).replace(html)
                await flyte.report.flush.aio()
            if config.on_error == "raise":
                raise RuntimeError(f"Pandera validation failed: {errors}")
            if config.on_error == "warn":
                logger.warning(str(errors))
                return data
            raise ValueError(f"Unsupported ValidationConfig.on_error value: {config.on_error}")
        if emit_report:
            html = self._report_renderer.to_html(title=report_title, data=validated, schema=schema)
            flyte.report.get_tab(report_title).replace(html)
            await flyte.report.flush.aio()
        return validated


def _all_pyspark_sql_typing_dataframe_classes() -> list[type[Any]]:
    """Every distinct ``DataFrame`` class from ``pandera.typing.pyspark_sql`` (handles duplicate loads)."""
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
        mod = importlib.import_module("pandera.typing.pyspark_sql")
        _add(getattr(mod, "DataFrame", None))
    except (ImportError, AttributeError):
        pass
    cached = sys.modules.get("pandera.typing.pyspark_sql")
    if cached is not None:
        _add(getattr(cached, "DataFrame", None))
    return out


def register_pandera_pyspark_sql_type_transformers() -> None:
    """Register Pandera validation + parquet I/O for ``pandera.typing.pyspark_sql.DataFrame``.

    Parquet encode/decode for ``pyspark.sql.DataFrame`` must be registered on
    ``DataFrameTransformerEngine`` (typically via ``flyteplugins-spark``).
    """
    classes = _all_pyspark_sql_typing_dataframe_classes()
    try:
        import pandera.typing.pyspark_sql as _pt  # noqa: F401
    except ImportError:
        pass
    classes = list(dict.fromkeys([*classes, *_all_pyspark_sql_typing_dataframe_classes()]))

    if not classes:
        logger.warning(
            "flyteplugins-pandera: could not resolve pandera.typing.pyspark_sql.DataFrame; "
            "is pandera with the pyspark extra installed?"
        )
        return

    transformer = PanderaPySparkSqlDataFrameTransformer()
    for cls in classes:
        TypeEngine.register_additional_type(transformer, cls, override=True)
        logger.debug("Registered PanderaPySparkSqlDataFrameTransformer for %r", cls)

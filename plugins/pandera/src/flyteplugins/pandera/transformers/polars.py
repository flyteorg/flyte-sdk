from __future__ import annotations

import importlib
import sys
import typing
from typing import Any

from flyte import system_logger as logger
from flyte.extend import lazy_module
from flyte.types import TypeEngine, TypeTransformerFailedError

from flyteplugins.pandera.renderers.polars import PanderaPolarsReportRenderer
from flyteplugins.pandera.transformers.base import PanderaDataFrameTransformer

if typing.TYPE_CHECKING:
    import polars as pl

    import pandera.typing.polars as pt
else:
    pl = lazy_module("polars")
    pt = lazy_module("pandera.typing.polars")


class PanderaPolarsDataFrameTransformer(PanderaDataFrameTransformer[Any]):
    _report_renderer: PanderaPolarsReportRenderer = PanderaPolarsReportRenderer()

    def __init__(self) -> None:
        super().__init__("Pandera Polars Transformer", pt.DataFrame)
        self._validation_memo: set[tuple[str, str]] = set()

    def _resolve_native_df_type(self, pandera_type: Any) -> type[Any]:
        origin = typing.get_origin(pandera_type) or pandera_type
        mod = getattr(origin, "__module__", "")
        name = getattr(origin, "__name__", "")
        if mod.endswith("typing.polars"):
            if name == "DataFrame":
                return pl.DataFrame
            if name == "LazyFrame":
                return pl.LazyFrame
        raise TypeTransformerFailedError(
            f"Only pandera.typing.polars.DataFrame and LazyFrame are supported (got {origin!r})."
        )


def register_pandera_polars_type_transformers() -> None:
    """Register one transformer for every distinct ``pandera.typing.polars`` ``DataFrame`` / ``LazyFrame`` class."""

    def _distinct_typing_container_classes() -> list[type[Any]]:
        """Resolve distinct DataFrame/LazyFrame classes from pandera.typing.polars (handles duplicate module loads)."""
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
            mod = importlib.import_module("pandera.typing.polars")
            _add(getattr(mod, "DataFrame", None))
            _add(getattr(mod, "LazyFrame", None))
        except (ImportError, AttributeError):
            pass
        cached = sys.modules.get("pandera.typing.polars")
        if cached is not None:
            _add(getattr(cached, "DataFrame", None))
            _add(getattr(cached, "LazyFrame", None))
        return out

    classes = _distinct_typing_container_classes()
    try:
        import pandera.typing.polars as _pt  # noqa: F401
    except ImportError:
        pass
    classes = list(dict.fromkeys([*classes, *_distinct_typing_container_classes()]))

    if not classes:
        logger.warning(
            "flyteplugins-pandera: could not resolve pandera.typing.polars DataFrame/LazyFrame; "
            "is pandera with the polars extra installed?"
        )
        return

    transformer = PanderaPolarsDataFrameTransformer()
    for cls in classes:
        TypeEngine.register_additional_type(transformer, cls, override=True)
        logger.debug("Registered PanderaPolarsDataFrameTransformer for %r", cls)

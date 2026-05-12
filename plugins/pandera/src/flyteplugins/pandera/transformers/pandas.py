from __future__ import annotations

import importlib
import sys
import typing
from typing import Any

from flyte._logging import logger
from flyte.extend import lazy_module
from flyte.types import TypeEngine, TypeTransformerFailedError

from flyteplugins.pandera.renderers.base import PanderaReportRenderer
from flyteplugins.pandera.renderers.pandas import PanderaPandasReportRenderer
from flyteplugins.pandera.transformers.base import PanderaDataFrameTransformer

if typing.TYPE_CHECKING:
    import pandas as pd

    import pandera.typing.pandas as pt
else:
    pd = lazy_module("pandas")
    pt = lazy_module("pandera.typing.pandas")


class PanderaPandasDataFrameTransformer(PanderaDataFrameTransformer[pt.DataFrame]):
    _report_renderer: PanderaReportRenderer = PanderaPandasReportRenderer()

    def __init__(self) -> None:
        super().__init__("Pandera DataFrame Transformer", pt.DataFrame)
        self._validation_memo: set[tuple[str, str]] = set()

    def _resolve_native_df_type(self, pandera_type: Any) -> type[pt.DataFrame]:
        origin = typing.get_origin(pandera_type) or pandera_type
        mod = getattr(origin, "__module__", "")
        name = getattr(origin, "__name__", "")
        if mod.endswith("typing.pandas") and name == "DataFrame":
            return pd.DataFrame
        raise TypeTransformerFailedError(f"Only pandera.typing.pandas.DataFrame is supported (got {origin!r}).")


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


def register_pandera_pandas_type_transformers() -> None:
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

    transformer = PanderaPandasDataFrameTransformer()
    for cls in classes:
        TypeEngine.register_additional_type(transformer, cls, override=True)
        logger.debug("Registered PanderaPandasDataFrameTransformer for %r", cls)

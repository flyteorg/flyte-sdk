from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from flyte._utils import lazy_module

from pandera.errors import SchemaErrors

from .pandas import DATA_PREVIEW_HEAD, PanderaPandasReportRenderer

if TYPE_CHECKING:
    import polars as pl
else:
    pl = lazy_module("polars")


class PanderaPolarsReportRenderer(PanderaPandasReportRenderer):
    """Great Tables reports for Polars containers (collects ``LazyFrame`` for previews)."""

    @staticmethod
    def _failure_cases_to_pandas(failure_cases: Any) -> Any:
        """Pandera's Polars backend attaches Polars ``DataFrame`` failure cases; pandas merge expects pandas."""
        if failure_cases is None:
            return None
        if isinstance(failure_cases, pl.LazyFrame):
            failure_cases = failure_cases.collect()
        if isinstance(failure_cases, pl.DataFrame):
            return failure_cases.to_pandas()
        return failure_cases

    def _create_error_report(self, data: Any, schema: Any, error: SchemaErrors):
        err = copy.copy(error)
        err.failure_cases = self._failure_cases_to_pandas(error.failure_cases)
        return super()._create_error_report(data, schema, err)

    @staticmethod
    def _to_pandas(data: Any):
        if isinstance(data, pl.LazyFrame):
            data = data.head(DATA_PREVIEW_HEAD).collect()
        if isinstance(data, pl.DataFrame):
            return data.head(DATA_PREVIEW_HEAD).to_pandas()
        return PanderaPandasReportRenderer._to_pandas(data)

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from flyte.extend import lazy_module

from flyteplugins.pandera.renderers.pandas import DATA_PREVIEW_HEAD, FAILURE_CASE_LIMIT, PanderaPandasReportRenderer
from pandera.errors import SchemaErrors

if TYPE_CHECKING:
    import pyspark.sql as psql
else:
    psql = lazy_module("pyspark.sql")


class PanderaPySparkSqlReportRenderer(PanderaPandasReportRenderer):
    """Great Tables reports for PySpark SQL ``DataFrame`` (preview via limited ``toPandas()``)."""

    @staticmethod
    def _failure_cases_to_pandas(failure_cases: Any) -> Any:
        if failure_cases is None:
            return None
        if isinstance(failure_cases, psql.DataFrame):
            return failure_cases.limit(FAILURE_CASE_LIMIT).toPandas()
        return failure_cases

    def _create_error_report(self, data: Any, schema: Any, error: SchemaErrors):
        err = copy.copy(error)
        err.failure_cases = self._failure_cases_to_pandas(error.failure_cases)
        return super()._create_error_report(data, schema, err)

    @staticmethod
    def _to_pandas(data: Any):
        if isinstance(data, psql.DataFrame):
            try:
                return data.limit(DATA_PREVIEW_HEAD).toPandas()
            except Exception:
                return None
        return PanderaPandasReportRenderer._to_pandas(data)

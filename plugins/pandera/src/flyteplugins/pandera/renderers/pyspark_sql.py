from __future__ import annotations

import copy
from typing import Any

from pandera.errors import SchemaErrors

from .pandas import DATA_PREVIEW_HEAD, FAILURE_CASE_LIMIT, PanderaPandasReportRenderer


def _pyspark_sql_types():
    import pyspark.sql as psql

    return psql


class PanderaPySparkSqlReportRenderer(PanderaPandasReportRenderer):
    """Great Tables reports for PySpark SQL ``DataFrame`` (preview via limited ``toPandas()``)."""

    @staticmethod
    def _failure_cases_to_pandas(failure_cases: Any) -> Any:
        if failure_cases is None:
            return None
        psql = _pyspark_sql_types()
        if isinstance(failure_cases, psql.DataFrame):
            return failure_cases.limit(FAILURE_CASE_LIMIT).toPandas()
        return failure_cases

    def _create_error_report(self, data: Any, schema: Any, error: SchemaErrors):
        err = copy.copy(error)
        err.failure_cases = self._failure_cases_to_pandas(error.failure_cases)
        return super()._create_error_report(data, schema, err)

    @staticmethod
    def _to_pandas(data: Any):
        psql = _pyspark_sql_types()
        if isinstance(data, psql.DataFrame):
            try:
                return data.limit(DATA_PREVIEW_HEAD).toPandas()
            except Exception:
                return None
        return PanderaPandasReportRenderer._to_pandas(data)

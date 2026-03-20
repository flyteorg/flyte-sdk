from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flyte._utils import lazy_module

if TYPE_CHECKING:
    import pandas as pd
else:
    pd = lazy_module("pandas")


@dataclass
class ValidationReport:
    summary: "pd.DataFrame"
    data_preview: "pd.DataFrame | None"
    failure_cases: "pd.DataFrame | None"


class PanderaReportRenderer:
    def __init__(self, title: str = "Pandera Validation Report"):
        self.title = title

    @staticmethod
    def _schema_name(schema: Any) -> str:
        return schema.name or getattr(schema, "__class__", type(schema)).__name__

    @staticmethod
    def _to_pandas_preview(data: Any) -> "pd.DataFrame | None":
        # This is a best-effort preview for mixed dataframe backends.
        if isinstance(data, pd.DataFrame):
            return data.head(10)
        if hasattr(data, "head"):
            try:
                head = data.head(10)
                if hasattr(head, "to_pandas"):
                    return head.to_pandas()
                if isinstance(head, pd.DataFrame):
                    return head
            except Exception:
                return None
        if hasattr(data, "to_pandas"):
            try:
                return data.to_pandas().head(10)
            except Exception:
                return None
        return None

    @staticmethod
    def _to_html_table(df: "pd.DataFrame | None") -> str:
        if df is None:
            return "<p><em>Not available for this dataframe backend.</em></p>"
        if df.empty:
            return "<p><em>None</em></p>"
        return df.to_html(index=False, escape=True)

    def _build_success(self, data: Any, schema: Any) -> ValidationReport:
        preview = self._to_pandas_preview(data)
        rows = preview.shape[0] if preview is not None else "unknown"
        cols = preview.shape[1] if preview is not None else "unknown"
        summary = pd.DataFrame(
            [
                {"field": "status", "value": "success"},
                {"field": "schema", "value": self._schema_name(schema)},
                {"field": "shape", "value": f"{rows} rows x {cols} columns"},
                {"field": "total_errors", "value": 0},
            ]
        )
        return ValidationReport(summary=summary, data_preview=preview, failure_cases=None)

    def _build_error(self, data: Any, schema: Any, error: Exception) -> ValidationReport:
        preview = self._to_pandas_preview(data)
        failure_cases = getattr(error, "failure_cases", None)
        if failure_cases is not None and not isinstance(failure_cases, pd.DataFrame):
            try:
                failure_cases = pd.DataFrame(failure_cases)
            except Exception:
                failure_cases = None

        total_errors = 1
        if isinstance(failure_cases, pd.DataFrame):
            total_errors = len(failure_cases.index)

        summary = pd.DataFrame(
            [
                {"field": "status", "value": "failed"},
                {"field": "schema", "value": self._schema_name(schema)},
                {"field": "error_type", "value": error.__class__.__name__},
                {"field": "total_errors", "value": total_errors},
                {"field": "message", "value": str(error)},
            ]
        )
        return ValidationReport(summary=summary, data_preview=preview, failure_cases=failure_cases)

    def to_html(self, data: Any, schema: Any, error: Exception | None = None) -> str:
        report = self._build_success(data, schema) if error is None else self._build_error(data, schema, error)
        status = "Validation succeeded" if error is None else "Validation failed"
        failure_html = ""
        if error is not None:
            failure_html = f"""
            <h2>Failure Cases</h2>
            {self._to_html_table(report.failure_cases)}
            """

        return f"""
        <div style="font-family: Arial, sans-serif; margin: 16px;">
          <h1>{self.title}</h1>
          <p><strong>{status}</strong></p>
          <h2>Summary</h2>
          {self._to_html_table(report.summary)}
          <h2>Data Preview</h2>
          {self._to_html_table(report.data_preview)}
          {failure_html}
        </div>
        """

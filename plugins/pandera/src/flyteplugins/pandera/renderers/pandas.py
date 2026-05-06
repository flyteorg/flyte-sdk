from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from flyte._utils import lazy_module

from .base import PanderaReportRenderer

if TYPE_CHECKING:
    import great_tables as gt
    import pandas
else:
    gt = lazy_module("great_tables")
    pandas = lazy_module("pandas")


from pandera.errors import SchemaErrors


@dataclass
class PandasReport:
    summary: pandas.DataFrame
    data_preview: pandas.DataFrame
    schema_error_df: pandas.DataFrame | None = None
    data_error_df: pandas.DataFrame | None = None


SCHEMA_ERROR_KEY = "SCHEMA"
DATA_ERROR_KEY = "DATA"
SCHEMA_ERROR_COLUMNS = ["schema", "column", "error_code", "check", "failure_case", "error"]
DATA_ERROR_COLUMNS = ["schema", "column", "error_code", "check", "index", "failure_case", "error"]
DATA_ERROR_DISPLAY_ORDER = ["column", "error_code", "percent_valid", "check", "failure_cases", "error"]

DATA_PREVIEW_HEAD = 5
FAILURE_CASE_LIMIT = 10
ERROR_COLUMN_MAX_WIDTH = 200


class PanderaPandasReportRenderer(PanderaReportRenderer):
    _FAILURE_CASES_TAIL_RE = re.compile(r"failure cases:\s*(.+)$", re.IGNORECASE | re.DOTALL)

    @staticmethod
    def _schema_name(schema: Any) -> str:
        if schema is None:
            return "unknown"
        return schema.name or getattr(schema, "__class__", type(schema)).__name__

    @classmethod
    def _extract_failure_case_text(cls, error_msg: Any) -> str:
        if error_msg is None:
            return ""
        text = str(error_msg).strip()
        if not text or text.lower() == "nan":
            return ""
        m = cls._FAILURE_CASES_TAIL_RE.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _to_pandas(data: Any) -> "pandas.DataFrame | None":
        if isinstance(data, pandas.DataFrame):
            return data
        if hasattr(data, "to_pandas"):
            try:
                return data.to_pandas()
            except Exception:
                return None
        return None

    def _create_success_report(self, data: "pandas.DataFrame", schema: Any) -> PandasReport:
        name = getattr(schema, "name", None) or self._schema_name(schema)
        summary = pandas.DataFrame(
            [
                {"Metadata": "Schema Name", "Value": name},
                {"Metadata": "Shape", "Value": f"{data.shape[0]} rows x {data.shape[1]} columns"},
                {"Metadata": "Total schema errors", "Value": 0},
                {"Metadata": "Total data errors", "Value": 0},
                {"Metadata": "Schema Object", "Value": f"```\n{schema!r}\n```"},
            ]
        )

        return PandasReport(
            summary=summary,
            data_preview=data.head(DATA_PREVIEW_HEAD),
            schema_error_df=None,
            data_error_df=None,
        )

    @staticmethod
    def _reshape_long_failure_cases(long_failure_cases: "pandas.DataFrame"):
        # Pandera can emit several ``column_in_dataframe`` rows that share the same
        # ``column`` cell (often the model class name) with different ``failure_case``
        # values (each missing field). ``pivot`` requires unique (index, columns).
        pivot_keys = ["schema_context", "check", "index", "column"]
        deduped = (
            long_failure_cases.groupby(pivot_keys, dropna=False, sort=False)["failure_case"]
            .agg(lambda s: ", ".join(s.dropna().astype(str).unique()))
            .reset_index()
        )
        return (
            deduped.pivot(index=["schema_context", "check", "index"], columns="column", values="failure_case")
            .apply(lambda s: s.to_dict(), axis="columns")
            .rename("failure_case")
            .reset_index(["index", "check"])
            .reset_index(drop=True)[["check", "index", "failure_case"]]
        )

    def _prepare_data_error_df_without_failure_cases(
        self, data: "pandas.DataFrame", data_errors: dict[str, Any]
    ) -> "pandas.DataFrame":
        """Build the data-error summary when ``failure_cases`` is missing (e.g. serialized error dict only)."""

        def num_failure_cases(series):
            return len(series)

        def _failure_cases(series):
            series = series.astype(str)
            out = ", ".join(str(x) for x in series.iloc[:FAILURE_CASE_LIMIT])
            if len(series) > FAILURE_CASE_LIMIT:
                out += f" ... (+{len(series) - FAILURE_CASE_LIMIT} more)"
            return out

        data_errors_df = pandas.concat(pandas.DataFrame(v).assign(error_code=k) for k, v in data_errors.items())
        extracted = data_errors_df["error"].map(self._extract_failure_case_text)
        data_errors_df = data_errors_df.assign(
            index=pandas.NA,
            failure_case=extracted.where(extracted != "", data_errors_df["error"].astype(str)),
        )
        for col in DATA_ERROR_COLUMNS:
            if col not in data_errors_df.columns:
                data_errors_df[col] = pandas.NA
        out_df = (
            data_errors_df[DATA_ERROR_COLUMNS]
            .groupby(["column", "error_code", "check", "error"])
            .failure_case.agg([num_failure_cases, _failure_cases])
            .reset_index()
            .rename(columns={"_failure_cases": "failure_cases"})
            .assign(percent_valid=lambda df: 1 - (df["num_failure_cases"] / data.shape[0]))
        )
        return out_df

    def _prepare_data_error_df(
        self,
        data: "pandas.DataFrame",
        data_errors: dict[str, Any],
        failure_cases: "pandas.DataFrame | None",
    ):
        if failure_cases is None:
            return self._prepare_data_error_df_without_failure_cases(data, data_errors)

        def num_failure_cases(series):
            return len(series)

        def _failure_cases(series):
            series = series.astype(str)
            out = ", ".join(str(x) for x in series.iloc[:FAILURE_CASE_LIMIT])
            if len(series) > FAILURE_CASE_LIMIT:
                out += f" ... (+{len(series) - FAILURE_CASE_LIMIT} more)"
            return out

        data_errors = pandas.concat(pandas.DataFrame(v).assign(error_code=k) for k, v in data_errors.items())
        long_failure_case_selector = (failure_cases["schema_context"] == "DataFrameSchema") & (
            failure_cases["column"].notna()
        )
        long_failure_cases = failure_cases[long_failure_case_selector]

        data_error_df = [data_errors.merge(failure_cases, how="inner", on=["column", "check"])]
        if long_failure_cases.shape[0] > 0:
            reshaped_failure_cases = self._reshape_long_failure_cases(long_failure_cases)
            long_data_errors = data_errors.assign(
                column=data_errors.column.where(~(data_errors.column == data_errors.schema), "NA")
            )
            data_error_df.append(
                long_data_errors.merge(reshaped_failure_cases, how="inner", on=["check"]).assign(column="NA")
            )

        data_error_df = pandas.concat(data_error_df)
        out_df = (
            data_error_df[DATA_ERROR_COLUMNS]
            .groupby(["column", "error_code", "check", "error"])
            .failure_case.agg([num_failure_cases, _failure_cases])
            .reset_index()
            .rename(columns={"_failure_cases": "failure_cases"})
            .assign(percent_valid=lambda df: 1 - (df["num_failure_cases"] / data.shape[0]))
        )
        return out_df

    def _create_error_report(
        self,
        data: "pandas.DataFrame",
        schema: Any,
        error: SchemaErrors | dict[str, Any],
    ):
        if isinstance(error, dict):
            failure_cases = None
            error_dict = error
            err_schema = schema
        else:
            failure_cases = getattr(error, "failure_cases", None)
            error_dict = error.message
            err_schema = getattr(error, "schema", schema)

        schema_errors = error_dict.get(SCHEMA_ERROR_KEY)
        data_errors = error_dict.get(DATA_ERROR_KEY)

        if schema_errors is None:
            schema_error_df = None
            total_schema_errors = 0
        else:
            schema_base = pandas.concat(pandas.DataFrame(v).assign(error_code=k) for k, v in schema_errors.items())
            if failure_cases is not None:
                schema_error_df = schema_base.merge(failure_cases, how="left", on=["column", "check"])[
                    SCHEMA_ERROR_COLUMNS
                ].drop(["schema"], axis="columns")
            else:
                if "error" in schema_base.columns:
                    extracted = schema_base["error"].map(self._extract_failure_case_text)
                    failure_col = extracted.where(extracted != "", schema_base["error"].astype(str))
                else:
                    failure_col = pandas.Series(pandas.NA, index=schema_base.index, dtype=object)
                schema_base = schema_base.assign(failure_case=failure_col)
                for col in SCHEMA_ERROR_COLUMNS:
                    if col not in schema_base.columns:
                        schema_base[col] = pandas.NA
                schema_error_df = schema_base[SCHEMA_ERROR_COLUMNS].drop(columns=["schema"], errors="ignore")
            total_schema_errors = schema_error_df.shape[0]

        if data_errors is None:
            data_error_df = None
            total_data_errors = 0
        else:
            data_error_df = self._prepare_data_error_df(data, data_errors, failure_cases)
            total_data_errors = data_error_df.shape[0]

        name = getattr(err_schema, "name", None) or self._schema_name(schema)
        summary = pandas.DataFrame(
            [
                {"Metadata": "Schema Name", "Value": name},
                {"Metadata": "Data Shape", "Value": f"{data.shape[0]} rows x {data.shape[1]} columns"},
                {"Metadata": "Total schema errors", "Value": total_schema_errors},
                {"Metadata": "Total data errors", "Value": total_data_errors},
                {"Metadata": "Schema Object", "Value": f"```\n{err_schema!r}\n```"},
            ]
        )

        return PandasReport(
            summary=summary,
            data_preview=data.head(DATA_PREVIEW_HEAD),
            schema_error_df=schema_error_df,
            data_error_df=data_error_df,
        )

    def _format_summary_df(self, df: "pandas.DataFrame") -> str:
        return (
            gt.GT(df)
            .tab_header(
                title=gt.md("**Summary**"),
                subtitle="A high-level overview of the schema errors found in the DataFrame.",
            )
            .cols_width(
                cases={
                    "Metadata": "20%",
                    "Value": "80%",
                }
            )
            .fmt_markdown(["Value"])
            .tab_stub(rowname_col="Metadata")
            .tab_stubhead(label="Metadata")
            .tab_style(style=gt.style.text(align="left"), locations=gt.loc.header())
            .tab_style(style=gt.style.fill(color="#f2fae2"), locations=gt.loc.header())
            .tab_style(
                style=gt.style.text(weight="bold"),
                locations=[gt.loc.column_labels(), gt.loc.stubhead(), gt.loc.stub()],
            )
            .as_raw_html()
        )

    def _format_data_preview_df(self, df: "pandas.DataFrame") -> str:
        return (
            gt.GT(df)
            .tab_header(
                title=gt.md("**Data Preview**"),
                subtitle=f"A preview of the first {min(DATA_PREVIEW_HEAD, df.shape[0])} rows of the data.",
            )
            .tab_style(style=gt.style.text(align="left"), locations=gt.loc.header())
            .tab_style(style=gt.style.fill(color="#f2fae2"), locations=gt.loc.header())
            .tab_style(style=gt.style.text(weight="bold"), locations=gt.loc.column_labels())
            .tab_style(style=gt.style.text(align="left"), locations=[gt.loc.body(), gt.loc.column_labels()])
            .as_raw_html()
        )

    @staticmethod
    def _format_error(x: str) -> str:
        if len(x) > ERROR_COLUMN_MAX_WIDTH:
            x = f"{x[:ERROR_COLUMN_MAX_WIDTH]}..."
        return f"```\n{x}\n```"

    def _format_schema_error_df(self, df: "pandas.DataFrame") -> str:
        df = df.assign(
            error=lambda df: df["error"].map(self._format_error),
            error_code=lambda df: df["error_code"].map(lambda x: f"`{x}`"),
            check=lambda df: df["check"].map(lambda x: f"`{x}`"),
        )
        return (
            gt.GT(df)
            .tab_header(
                title=gt.md("**Schema-level Errors**"),
                subtitle="Schema-level metadata errors, e.g. column names, dtypes.",
            )
            .fmt_markdown(["error_code", "check", "error"])
            .tab_style(style=gt.style.text(align="left"), locations=gt.loc.header())
            .tab_style(style=gt.style.fill(color="#f2fae2"), locations=gt.loc.header())
            .tab_style(style=gt.style.text(weight="bold"), locations=gt.loc.column_labels())
            .as_raw_html()
        )

    def _format_data_error_df(self, df: "pandas.DataFrame") -> str:
        df = df.assign(
            error=lambda df: df["error"].map(self._format_error),
            error_code=lambda df: df["error_code"].map(lambda x: f"`{x}`"),
            check=lambda df: df["check"].map(lambda x: f"`{x}`"),
        )[DATA_ERROR_DISPLAY_ORDER]

        return (
            gt.GT(df)
            .tab_header(
                title=gt.md("**Data-level Errors**"),
                subtitle="Data-level value errors, e.g. null values, out-of-range values.",
            )
            .fmt_markdown(["error_code", "check", "error"])
            .fmt_percent("percent_valid", decimals=2)
            .data_color(columns=["percent_valid"], palette="RdYlGn", domain=[0, 1], alpha=0.2)
            .tab_stub(groupname_col="column", rowname_col="error_code")
            .tab_stubhead(label="column")
            .tab_style(
                style=gt.style.text(align="left"), locations=[gt.loc.header(), gt.loc.column_labels(), gt.loc.body()]
            )
            .tab_style(style=gt.style.text(align="center"), locations=gt.loc.body(columns="percent_valid"))
            .tab_style(style=gt.style.fill(color="#f2fae2"), locations=gt.loc.header())
            .tab_style(
                style=gt.style.text(weight="bold"),
                locations=[gt.loc.column_labels(), gt.loc.stubhead(), gt.loc.row_groups()],
            )
            .tab_style(
                style=gt.style.fill(color="#f4f4f4"),
                locations=[gt.loc.row_groups(), gt.loc.stub()],
            )
            .as_raw_html()
        )

    def _format_generic_failure_df(self, df: "pandas.DataFrame") -> str:
        return (
            gt.GT(df)
            .tab_header(
                title=gt.md("**Failure Cases**"),
                subtitle="Raw failure-case rows attached to the validation error, when available.",
            )
            .tab_style(style=gt.style.text(align="left"), locations=gt.loc.header())
            .tab_style(style=gt.style.fill(color="#f2fae2"), locations=gt.loc.header())
            .tab_style(style=gt.style.text(weight="bold"), locations=gt.loc.column_labels())
            .as_raw_html()
        )

    def _generic_exception_report(self, data: Any, schema: Any, error: Exception) -> str:
        preview = self._to_pandas(data)
        if preview is not None:
            preview = preview.head(DATA_PREVIEW_HEAD)
        failure_cases = getattr(error, "failure_cases", None)
        if failure_cases is not None and not isinstance(failure_cases, pandas.DataFrame):
            try:
                failure_cases = pandas.DataFrame(failure_cases)
            except Exception:
                failure_cases = None

        total_errors = 1
        if isinstance(failure_cases, pandas.DataFrame):
            total_errors = len(failure_cases.index)

        summary = pandas.DataFrame(
            [
                {"Metadata": "Status", "Value": "failed"},
                {"Metadata": "Schema", "Value": self._schema_name(schema)},
                {"Metadata": "Error type", "Value": error.__class__.__name__},
                {"Metadata": "Total errors", "Value": str(total_errors)},
                {"Metadata": "Message", "Value": f"```\n{error!s}\n```"},
            ]
        )
        if preview is None:
            preview = pandas.DataFrame([{"note": "Not available (need a pandas-compatible dataframe)."}])

        error_segments = ""
        if isinstance(failure_cases, pandas.DataFrame) and not failure_cases.empty:
            error_segments = f"""
                <br>
                {self._format_generic_failure_df(failure_cases)}
                """

        return f"""
                {self._format_summary_df(summary)}
                <br>
                {self._format_data_preview_df(preview)}
                {error_segments}
                """

    def to_html(
        self,
        title: str,
        data: Any,
        schema: Any,
        error: SchemaErrors | dict[str, Any] | None = None,
        warn: bool = False,
    ) -> str:
        if hasattr(schema, "to_schema"):
            schema = schema.to_schema()

        df = self._to_pandas(data)
        error_segments = ""

        if error is None:
            if df is None:
                placeholder = pandas.DataFrame([{"note": "Not available (need a pandas-compatible dataframe)."}])
                summary = pandas.DataFrame(
                    [
                        {"Metadata": "Status", "Value": "success (preview only)"},
                        {"Metadata": "Schema", "Value": self._schema_name(schema)},
                    ]
                )
                inner = f"""
                {self._format_summary_df(summary)}
                <br>
                {self._format_data_preview_df(placeholder)}
                """
            else:
                report_dfs = self._create_success_report(df, schema)
                inner = f"""
                {self._format_summary_df(report_dfs.summary)}
                <br>
                {self._format_data_preview_df(report_dfs.data_preview)}
                """
            top_message = "✅ Data validation succeeded."
        else:
            icon = "⚠️" if warn else "❌"
            if df is not None and (
                (isinstance(error, dict) and (SCHEMA_ERROR_KEY in error or DATA_ERROR_KEY in error))
                or (SchemaErrors and isinstance(error, SchemaErrors))
            ):
                report_dfs = self._create_error_report(df, schema, error)
                top_message = f"{icon} Data validation failed."
                inner = f"""
                {self._format_summary_df(report_dfs.summary)}
                <br>
                {self._format_data_preview_df(report_dfs.data_preview)}
                """
                if report_dfs.schema_error_df is not None:
                    error_segments += f"""
                    <br>
                    {self._format_schema_error_df(report_dfs.schema_error_df)}
                    """
                if report_dfs.data_error_df is not None:
                    error_segments += f"""
                    <br>
                    {self._format_data_error_df(report_dfs.data_error_df)}
                    """
            else:
                top_message = f"{icon} Data validation failed."
                inner = self._generic_exception_report(data, schema, error)

        bootstrap_css = "https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css"
        bootstrap_integrity = "sha384-HSMxcRTRxnN+Bdg0JdbxYKrThecOKuH5zCYotlSAcp1+c8xmyTe9GYg1l9a69psu"
        pandera_logo = "https://raw.githubusercontent.com/pandera-dev/pandera/main/docs/source/_static/pandera-logo.png"
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <link rel="stylesheet" href="{bootstrap_css}" integrity="{bootstrap_integrity}" crossorigin="anonymous">
            <style>
                .pandera-report {{
                    min-width: 60%;
                    margin: 0 auto;
                    padding: 20px;
                }}

                .report-title h1 {{
                    display: inline;
                    line-height: 60px;
                    padding-left: 10px;
                }}

                .report-title img {{
                    display: inline;
                    width: 60px;
                    float: left;
                }}

                table.gt_table {{
                    width: 100% !important;
                    margin-left: 0 !important;
                }}
            </style>
        </head>
        <body>
            <div class="pandera-report">
                <div class="report-title">
                    <img src="{pandera_logo}" alt="Pandera Logo">
                    <h1>{title}</h1>
                </div>

                <h3>{top_message}</h3>

                {inner}
                {error_segments}
            </div>
        </body>
        </html>
        """

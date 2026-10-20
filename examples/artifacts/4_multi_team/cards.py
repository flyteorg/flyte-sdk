"""Artifact cards for the example: what is in each published version, readable in the console.

Every artifact the example publishes carries one. A card is a self-contained HTML page attached to the
version (`artifacts.Card.create_from(...)`, then `handle.at(..., card=card)`), shown on the artifact's
"Artifact Card" tab. They all share one layout so the graph reads as one product:

- a header: the artifact, its partition, and what it is
- a strip of headline numbers
- the body: schema and a preview for data, metrics for a model, the report itself for a report
- a footer: which task built it, from what

Nothing here imports pandas at module level, so the seed task (a slim image) can use it too.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

_STYLE = """
:root { color-scheme: light dark; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --bg:#fff; --chip:#f2f2f7;
        --accent:#e8a33d; --good:#34a853; --bad:#d93025; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#f5f5f7; --muted:#a1a1a6; --line:#3a3a3c; --bg:#1c1c1e; --chip:#2c2c2e; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.card { max-width: 920px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin: 24px 0 8px; }
p.desc { color: var(--muted); margin: 0 0 12px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin: 8px 0 0; }
.chip { background:var(--chip); border-radius:999px; padding:2px 10px;
        font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
.kind { background: var(--accent); color:#1d1d1f; }
.stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:12px; margin-top:16px; }
.stat { border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.stat b { display:block; font-size:22px; }
.stat span { color:var(--muted); font-size:12px; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:10px; }
table { border-collapse: collapse; width:100%; font-size: 13px; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--muted); font-weight:600; }
tr:last-child td { border-bottom: 0; }
td.num { text-align:right; font-variant-numeric: tabular-nums; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.bar { height:8px; border-radius:4px; background:var(--accent); min-width:2px; }
.track { width:140px; background:var(--chip); border-radius:4px; }
.footer { margin-top:24px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
.report { border:1px solid var(--line); border-radius:10px; padding:16px; }
"""


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value and abs(value) < 0.01:
            return f"{value:.3g}"  # a learning rate is 0.0003, not 0
        return f"{value:,.3f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:,.0f}"
    return _e(value)


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]], numeric: Sequence[int] = ()) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(f'<td class="num">{_num(v)}</td>' if i in numeric else f"<td>{_e(v)}</td>" for i, v in enumerate(row))
        + "</tr>"
        for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _page(
    *,
    name: str,
    kind: str,
    description: str,
    partition: Mapping[str, Any],
    stats: Sequence[Tuple[str, Any]],
    body: str,
    built_by: str,
) -> str:
    chips = [f'<span class="chip kind">{_e(kind)}</span>'] + [
        f'<span class="chip">{_e(k)}={_e(v)}</span>' for k, v in partition.items()
    ]
    stat_html = "".join(f'<div class="stat"><b>{_num(v)}</b><span>{_e(k)}</span></div>' for k, v in stats)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{_STYLE}</style></head><body><div class='card'>"
        f"<h1 class='mono'>{_e(name)}</h1><p class='desc'>{_e(description)}</p>"
        f"<div class='chips'>{''.join(chips)}</div>"
        f"<div class='stats'>{stat_html}</div>"
        f"{body}"
        f"<div class='footer'>{built_by}</div>"
        "</div></body></html>"
    )


def _built_by(task: str, inputs: Sequence[str]) -> str:
    ins = ", ".join(f"<span class='mono'>{_e(i)}</span>" for i in inputs)
    return f"Built by <span class='mono'>{_e(task)}</span>" + (f" from {ins}" if ins else "") + "."


def dataframe_card(
    df: Any,
    *,
    name: str,
    description: str,
    partition: Mapping[str, Any],
    task: str,
    inputs: Sequence[str] = (),
    preview_rows: int = 10,
) -> str:
    """A data card for a pandas DataFrame: size, schema, numeric ranges, and the first rows."""
    import pandas as pd

    rows, cols = df.shape
    nulls = int(df.isna().sum().sum())
    schema = []
    for c in df.columns:
        s = df[c]
        example = s.dropna().iloc[0] if s.notna().any() else ""
        schema.append((c, str(s.dtype), int(s.notna().sum()), int(s.nunique(dropna=True)), example))
    numeric = df.select_dtypes("number")
    ranges = ""
    if not numeric.empty:
        peak = max((float(numeric[c].abs().max() or 0) for c in numeric.columns), default=0.0) or 1.0
        bars = []
        for c in numeric.columns:
            s = numeric[c]
            mean = float(s.mean()) if len(s) else 0.0
            width = max(2, int(140 * abs(mean) / peak))
            bars.append(
                f"<tr><td class='mono'>{_e(c)}</td><td class='num'>{_num(float(s.min()) if len(s) else 0)}</td>"
                f"<td class='num'>{_num(mean)}</td><td class='num'>{_num(float(s.max()) if len(s) else 0)}</td>"
                f"<td><div class='track'><div class='bar' style='width:{width}px'></div></div></td></tr>"
            )
        ranges = (
            "<h2>Numeric columns</h2><div class='scroll'><table><thead><tr><th>column</th><th>min</th>"
            f"<th>mean</th><th>max</th><th>mean, relative</th></tr></thead><tbody>{''.join(bars)}</tbody></table></div>"
        )
    head = df.head(preview_rows)
    preview = _table(
        [str(c) for c in head.columns],
        head.itertuples(index=False, name=None),
        numeric=[i for i, c in enumerate(head.columns) if pd.api.types.is_numeric_dtype(head[c])],
    )
    body = (
        "<h2>Schema</h2>"
        + _table(["column", "type", "non-null", "distinct", "example"], schema, numeric=[2, 3])
        + ranges
        + f"<h2>First {min(preview_rows, rows)} of {rows:,} rows</h2>"
        + preview
    )
    return _page(
        name=name,
        kind="data",
        description=description,
        partition=partition,
        stats=[("rows", rows), ("columns", cols), ("null values", nulls)],
        body=body,
        built_by=_built_by(task, inputs),
    )


def csv_card(
    text: str,
    *,
    name: str,
    description: str,
    partition: Mapping[str, Any],
    task: str,
    preview_rows: int = 10,
) -> str:
    """A data card for a raw CSV file, read with the standard library (no pandas in the image)."""
    import csv
    import io

    reader = list(csv.reader(io.StringIO(text)))
    header: List[str] = reader[0] if reader else []
    data = reader[1:]
    schema = []
    for i, col in enumerate(header):
        values = [r[i] for r in data if i < len(r) and r[i] != ""]
        numeric = all(_is_number(v) for v in values) and bool(values)
        schema.append(
            (col, "number" if numeric else "text", len(values), len(set(values)), values[0] if values else "")
        )
    body = (
        "<h2>Columns</h2>"
        + _table(["column", "type", "non-empty", "distinct", "example"], schema, numeric=[2, 3])
        + f"<h2>First {min(preview_rows, len(data))} of {len(data):,} rows</h2>"
        + _table(header, data[:preview_rows])
    )
    return _page(
        name=name,
        kind="data",
        description=description,
        partition=partition,
        stats=[("rows", len(data)), ("columns", len(header)), ("bytes", len(text.encode()))],
        body=body,
        built_by=_built_by(task, []),
    )


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def model_card(
    weights: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    name: str,
    description: str,
    partition: Mapping[str, Any],
    task: str,
    inputs: Sequence[str] = (),
) -> str:
    """A model card: training metrics, and each feature's mean for churned vs retained users."""
    churned: Mapping[str, float] = weights.get("churned", {})
    retained: Mapping[str, float] = weights.get("retained", {})
    feats = sorted(set(churned) | set(retained))
    peak = max([abs(float(v)) for v in list(churned.values()) + list(retained.values())] or [1.0]) or 1.0

    def bar(value: float, color: str) -> str:
        width = max(2, int(140 * abs(value) / peak))
        return (
            f"<td class='num'>{_num(value)}</td><td><div class='track'>"
            f"<div class='bar' style='width:{width}px;background:var({color})'></div></div></td>"
        )

    rows = [
        f"<tr><td class='mono'>{_e(f)}</td>"
        + bar(float(churned.get(f, 0.0)), "--bad")
        + bar(float(retained.get(f, 0.0)), "--good")
        + "</tr>"
        for f in feats
    ]
    body = (
        "<h2>Feature means: churned vs retained</h2><div class='scroll'><table><thead><tr><th>feature</th>"
        "<th>churned</th><th></th><th>retained</th><th></th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<h2>Training</h2>"
        + _table(["metric", "value"], [*sorted(metrics.items()), ("learning rate", weights.get("lr", ""))], numeric=[1])
    )
    churn_rate = metrics.get("churn_rate")
    return _page(
        name=name,
        kind="model",
        description=description,
        partition=partition,
        stats=[
            ("training rows", metrics.get("rows", 0)),
            ("days in window", metrics.get("days", 0)),
            ("churn rate", f"{100 * float(churn_rate):.1f}%" if churn_rate is not None else "-"),
        ],
        body=body,
        built_by=_built_by(task, inputs),
    )


def report_card(
    report_html: str,
    *,
    name: str,
    description: str,
    partition: Mapping[str, Any],
    stats: Sequence[Tuple[str, Any]],
    task: str,
    inputs: Sequence[str] = (),
    at_risk: Optional[Sequence[Tuple[str, Any]]] = None,
) -> str:
    """A report card: the headline numbers, the report itself, and who is at risk."""
    body = "<h2>Report</h2><div class='report'>" + report_html + "</div>"
    if at_risk:
        body += "<h2>Users at risk</h2>" + _table(["user", "buys in the window"], at_risk, numeric=[1])
    return _page(
        name=name,
        kind="data",
        description=description,
        partition=partition,
        stats=stats,
        body=body,
        built_by=_built_by(task, inputs),
    )

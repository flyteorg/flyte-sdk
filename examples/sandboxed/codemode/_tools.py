"""Tool functions and datasets for the Chat Analytics Agent.

To add a new tool:
1. Write a function with type annotations and a docstring.
2. Add it to ``ALL_TOOLS`` at the bottom of this file.

The agent auto-generates its system prompt from the signatures and docstrings
of every function in ``ALL_TOOLS``, so there is nothing else to update.
"""

from __future__ import annotations

import json as _json
import math
from typing import Callable

# ---------------------------------------------------------------------------
# Chart palette (Union golden palette — based on union.ai brand gold #e69812)
# ---------------------------------------------------------------------------

CHART_COLORS = [
    "rgba(230, 152, 18, 0.8)",  # #e69812 — primary gold
    "rgba(242, 189, 82, 0.8)",  # #f2bd52 — lighter amber
    "rgba(184, 119, 10, 0.8)",  # #b8770a — darker bronze
    "rgba(250, 210, 130, 0.8)",  # #fad282 — honey
    "rgba(140, 90, 5, 0.8)",  # #8c5a05 — deep gold
]

CHART_BORDERS = [
    "#e69812",
    "#f2bd52",
    "#b8770a",
    "#fad282",
    "#8c5a05",
]

# ---------------------------------------------------------------------------
# Datasets (returned by fetch_data)
# ---------------------------------------------------------------------------

_SALES_2024_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_SALES_2024_REGIONS = ["North", "South", "East", "West"]
_SALES_2024_BASE = {"North": 120000, "South": 95000, "East": 110000, "West": 105000}
_SALES_2024_SEASONAL = [0.85, 0.88, 0.95, 1.0, 1.05, 1.10, 1.08, 1.12, 1.06, 1.02, 1.15, 1.25]


def _build_sales_2024() -> list[dict]:
    rows: list[dict] = []
    for i, month in enumerate(_SALES_2024_MONTHS):
        for region in _SALES_2024_REGIONS:
            revenue = int(_SALES_2024_BASE[region] * _SALES_2024_SEASONAL[i])
            rows.append({"month": month, "region": region, "revenue": revenue, "units": revenue // 45})
    return rows


def _build_employees() -> list[dict]:
    return [
        {"name": "Alice", "department": "Engineering", "salary": 125000, "years_exp": 8, "performance_rating": 4.5},
        {"name": "Bob", "department": "Engineering", "salary": 110000, "years_exp": 5, "performance_rating": 3.8},
        {"name": "Carol", "department": "Marketing", "salary": 95000, "years_exp": 6, "performance_rating": 4.2},
        {"name": "Dave", "department": "Marketing", "salary": 88000, "years_exp": 3, "performance_rating": 3.5},
        {"name": "Eve", "department": "Sales", "salary": 92000, "years_exp": 4, "performance_rating": 4.7},
        {"name": "Frank", "department": "Sales", "salary": 105000, "years_exp": 7, "performance_rating": 4.0},
        {"name": "Grace", "department": "Engineering", "salary": 135000, "years_exp": 10, "performance_rating": 4.8},
        {"name": "Hank", "department": "HR", "salary": 78000, "years_exp": 2, "performance_rating": 3.9},
        {"name": "Ivy", "department": "HR", "salary": 82000, "years_exp": 4, "performance_rating": 4.1},
        {"name": "Jack", "department": "Sales", "salary": 98000, "years_exp": 5, "performance_rating": 3.6},
    ]


def _build_website_traffic() -> list[dict]:
    pages = ["Home", "Products", "Blog", "Pricing", "Docs"]
    rows: list[dict] = []
    base_visitors = {"Home": 5000, "Products": 3200, "Blog": 2800, "Pricing": 1500, "Docs": 2100}
    for day in range(1, 31):
        for page in pages:
            visitors = base_visitors[page] + (day * 17 % 400) - 200
            bounce = round(0.35 + (day % 7) * 0.03 + (pages.index(page) * 0.05), 2)
            duration = round(2.5 + visitors / 2000 - bounce, 1)
            rows.append(
                {
                    "date": f"2024-01-{day:02d}",
                    "page": page,
                    "visitors": max(visitors, 100),
                    "bounce_rate": min(bounce, 0.85),
                    "avg_duration": max(duration, 0.5),
                }
            )
    return rows


def _build_inventory() -> list[dict]:
    return [
        {"product": "Widget A", "category": "Electronics", "stock": 150, "price": 29.99, "supplier": "TechCo"},
        {"product": "Widget B", "category": "Electronics", "stock": 75, "price": 49.99, "supplier": "TechCo"},
        {"product": "Gadget X", "category": "Electronics", "stock": 200, "price": 19.99, "supplier": "GizmoInc"},
        {"product": "Chair Pro", "category": "Furniture", "stock": 30, "price": 299.99, "supplier": "OfficePlus"},
        {"product": "Desk Std", "category": "Furniture", "stock": 45, "price": 199.99, "supplier": "OfficePlus"},
        {"product": "Lamp LED", "category": "Furniture", "stock": 120, "price": 39.99, "supplier": "BrightLite"},
        {"product": "Notebook", "category": "Stationery", "stock": 500, "price": 4.99, "supplier": "PaperWorld"},
        {"product": "Pen Pack", "category": "Stationery", "stock": 350, "price": 7.99, "supplier": "PaperWorld"},
        {"product": "Headset Z", "category": "Electronics", "stock": 60, "price": 89.99, "supplier": "GizmoInc"},
        {"product": "Monitor 27", "category": "Electronics", "stock": 25, "price": 399.99, "supplier": "TechCo"},
    ]


_DATASETS: dict[str, Callable[[], list[dict]]] = {
    "sales_2024": _build_sales_2024,
    "employees": _build_employees,
    "website_traffic": _build_website_traffic,
    "inventory": _build_inventory,
}

# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def fetch_data(dataset: str) -> list:
    """Fetch tabular data by dataset name.

    Available datasets:
    - "sales_2024": columns month, region, revenue, units
    - "employees": columns name, department, salary, years_exp, performance_rating
    - "website_traffic": columns date, page, visitors, bounce_rate, avg_duration
    - "inventory": columns product, category, stock, price, supplier
    """
    builder = _DATASETS.get(dataset)
    if builder is not None:
        return builder()
    return []


def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    """Generate a self-contained Chart.js HTML snippet.

    Args:
        chart_type: One of "bar", "line", "pie", "doughnut".
        title: Chart title displayed above the canvas.
        labels: X-axis labels (or slice labels for pie/doughnut).
        values: Either a flat list of numbers, or a list of
                {"label": str, "data": list[number]} dicts for multi-series.

    Returns:
        HTML string with a <canvas> and <script> block.
    """
    canvas_id = "chart-" + title.lower().replace(" ", "-").replace("/", "-")

    if values and isinstance(values[0], dict):
        datasets = []
        for i, series in enumerate(values):
            color_idx = i % len(CHART_COLORS)
            datasets.append(
                {
                    "label": series["label"],
                    "data": series["data"],
                    "backgroundColor": CHART_COLORS[color_idx],
                    "borderColor": CHART_BORDERS[color_idx],
                    "borderWidth": 2,
                    "tension": 0.3,
                    "fill": False,
                }
            )
    else:
        bg_colors = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(values))]
        border_colors = [CHART_BORDERS[i % len(CHART_BORDERS)] for i in range(len(values))]
        datasets = [
            {
                "label": title,
                "data": values,
                "backgroundColor": bg_colors if chart_type in ("pie", "doughnut") else CHART_COLORS[0],
                "borderColor": border_colors if chart_type in ("pie", "doughnut") else CHART_BORDERS[0],
                "borderWidth": 2,
                "tension": 0.3,
                "fill": chart_type == "line",
            }
        ]

    config = {
        "type": chart_type,
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {"title": {"display": True, "text": title, "font": {"size": 16}}},
        },
    }

    return (
        f'<div style="position:relative;height:350px;margin:20px 0;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f"<script>new Chart(document.getElementById('{canvas_id}'),"
        f"{_json.dumps(config)});</script>"
    )


def calculate_statistics(data: list, column: str) -> dict:
    """Calculate descriptive statistics for a numeric column.

    Args:
        data: List of row dicts (e.g. from fetch_data).
        column: Name of the numeric column to analyze.

    Returns:
        Dict with keys: count, mean, median, min, max, std_dev.
    """
    vals = [row[column] for row in data if column in row]
    if not vals:
        return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}
    n = len(vals)
    total = sum(vals)
    mean = total / n
    sorted_vals = sorted(vals)
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    variance = sum((v - mean) ** 2 for v in vals) / n
    return {
        "count": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": min(vals),
        "max": max(vals),
        "std_dev": round(math.sqrt(variance), 2),
    }


def filter_data(data: list, column: str, operator: str, value: object) -> list:
    """Filter rows where *column* matches the condition.

    Args:
        data: List of row dicts.
        column: Column name to test.
        operator: One of "==", "!=", ">", ">=", "<", "<=".
        value: The value to compare against.

    Returns:
        Filtered list of row dicts.
    """
    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
    }
    fn = ops.get(operator)
    if fn is None:
        return data
    return [row for row in data if column in row and fn(row[column], value)]


def group_and_aggregate(data: list, group_by: str, agg_column: str, agg_func: str) -> list:
    """Group rows and aggregate a numeric column.

    Args:
        data: List of row dicts.
        group_by: Column to group on.
        agg_column: Numeric column to aggregate.
        agg_func: One of "sum", "mean", "count", "min", "max".

    Returns:
        List of {"group": key, "value": aggregated_value} dicts.
    """
    groups: dict[object, list[float]] = {}
    for row in data:
        key = row.get(group_by)
        if key is not None and agg_column in row:
            groups.setdefault(key, []).append(row[agg_column])

    results: list[dict] = []
    for key, vals in groups.items():
        if agg_func == "sum":
            agg = sum(vals)
        elif agg_func == "mean":
            agg = round(sum(vals) / len(vals), 2)
        elif agg_func == "count":
            agg = len(vals)
        elif agg_func == "min":
            agg = min(vals)
        elif agg_func == "max":
            agg = max(vals)
        else:
            agg = sum(vals)
        results.append({"group": key, "value": agg})
    return results


def sort_data(data: list, column: str, descending: bool = False) -> list:
    """Sort rows by a column.

    Args:
        data: List of row dicts.
        column: Column to sort by.
        descending: If True, sort in descending order.

    Returns:
        New sorted list of row dicts.
    """
    return sorted(data, key=lambda row: row.get(column, 0), reverse=descending)


# ---------------------------------------------------------------------------
# Registry — single source of truth for prompt generation and sandbox
# ---------------------------------------------------------------------------

ALL_TOOLS: dict[str, Callable] = {
    "fetch_data": fetch_data,
    "create_chart": create_chart,
    "calculate_statistics": calculate_statistics,
    "filter_data": filter_data,
    "group_and_aggregate": group_and_aggregate,
    "sort_data": sort_data,
}

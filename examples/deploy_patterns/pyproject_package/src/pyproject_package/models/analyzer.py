"""Data analysis utilities.

This module provides functions to analyze processed data and generate reports.
It uses numpy for numerical operations and has no Flyte dependencies.
"""

from typing import Any

import numpy as np


def calculate_statistics(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate basic statistics on the data.

    Args:
        data: List of data items with numerical values

    Returns:
        Dictionary containing statistical measures

    Example:
        >>> data = [{"value": 10.5}, {"value": 20.5}, {"value": 15.0}]
        >>> stats = calculate_statistics(data)
        >>> print(stats["mean"])
    """
    if not data:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    values = np.array([item["value"] for item in data])

    stats = {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std_dev": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentile_25": float(np.percentile(values, 25)),
        "percentile_75": float(np.percentile(values, 75)),
    }

    return stats


def generate_report(stats: dict[str, Any]) -> str:
    """Generate a formatted analysis report.

    Args:
        stats: Dictionary containing statistical measures and aggregations

    Returns:
        Formatted report string

    Example:
        >>> stats = {"basic": {"count": 10, "mean": 15.5}}
        >>> report = generate_report(stats)
        >>> print(report)
    """
    report_lines = [
        "=" * 60,
        "DATA ANALYSIS REPORT",
        "=" * 60,
    ]

    # Basic statistics section
    if "basic" in stats:
        basic = stats["basic"]
        report_lines.extend(
            [
                "",
                "BASIC STATISTICS:",
                f"  Count:       {basic.get('count', 0)}",
                f"  Mean:        {basic.get('mean', 0.0):.2f}",
                f"  Median:      {basic.get('median', 0.0):.2f}",
                f"  Std Dev:     {basic.get('std_dev', 0.0):.2f}",
                f"  Min:         {basic.get('min', 0.0):.2f}",
                f"  Max:         {basic.get('max', 0.0):.2f}",
                f"  25th %ile:   {basic.get('percentile_25', 0.0):.2f}",
                f"  75th %ile:   {basic.get('percentile_75', 0.0):.2f}",
            ]
        )

    # Category aggregations section
    if "aggregated" in stats and "categories" in stats["aggregated"]:
        categories = stats["aggregated"]["categories"]
        total_items = stats["aggregated"].get("total_items", 0)

        report_lines.extend(
            [
                "",
                "CATEGORY BREAKDOWN:",
                f"  Total Items: {total_items}",
                "",
            ]
        )

        for category, cat_stats in sorted(categories.items()):
            report_lines.extend(
                [
                    f"  Category: {category.upper()}",
                    f"    Count:         {cat_stats.get('count', 0)}",
                    f"    Total Value:   {cat_stats.get('total_value', 0.0):.2f}",
                    f"    Average Value: {cat_stats.get('average_value', 0.0):.2f}",
                    "",
                ]
            )

    report_lines.append("=" * 60)

    return "\n".join(report_lines)

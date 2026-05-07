"""Durable CodeModeAgent in a pure task context (no UI).

This mirrors the execution style of `examples/sandbox/codemode/durable_agent.py`,
but uses the SDK `flyte.ai.CodeModeAgent` and keeps everything in one file.

The agent generates Monty-safe Python code and executes it in a sandbox. Tool
calls dispatch as durable Flyte tasks because the tools are defined as
``@task_env.task``.

Run::

    flyte run examples/agents/codemode_durable_task_agent.py analyze \\
        --request "Show monthly revenue trends for sales_2024 by region"
"""

from __future__ import annotations

from typing import Any

import flyte
from flyte.ai import CodeModeAgent

task_env = flyte.TaskEnvironment(
    name="codemode-durable-task-agent",
    secrets=[flyte.Secret(key="internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY")],
    image=flyte.Image.from_debian_base().with_pip_packages("httpx", "pydantic-monty", "litellm"),
    reusable=flyte.ReusePolicy(replicas=1, concurrency=10),
)


def _build_sales_2024() -> list[dict[str, Any]]:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    regions = ["North", "South", "East", "West"]
    base = {"North": 120000, "South": 95000, "East": 110000, "West": 105000}
    seasonal = [0.85, 0.88, 0.95, 1.0, 1.05, 1.10, 1.08, 1.12, 1.06, 1.02, 1.15, 1.25]
    rows: list[dict[str, Any]] = []
    for i, month in enumerate(months):
        for region in regions:
            revenue = int(base[region] * seasonal[i])
            rows.append({"month": month, "region": region, "revenue": revenue, "units": revenue // 45})
    return rows


_DATASETS: dict[str, Any] = {
    "sales_2024": _build_sales_2024,
}


@task_env.task
async def fetch_data(dataset: str) -> list:
    """Fetch tabular data by dataset name.

    Available datasets:
    - "sales_2024": columns month, region, revenue, units
    """
    builder = _DATASETS.get(dataset)
    if builder is None:
        return []
    return builder()


@task_env.task
async def group_and_aggregate(data: list, group_by: str, agg_column: str, agg_func: str) -> list:
    """Group rows and aggregate a numeric column.

    Args:
        data: List of row dicts (e.g. from fetch_data).
        group_by: Column to group on.
        agg_column: Numeric column to aggregate.
        agg_func: One of "sum", "mean", "count", "min", "max".

    Returns:
        List of {"group": key, "value": aggregated_value} dicts.
    """
    groups: dict[Any, list[Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        k = row.get(group_by)
        v = row.get(agg_column)
        groups.setdefault(k, []).append(v)

    out: list[dict[str, Any]] = []
    for k, vals in groups.items():
        nums = [v for v in vals if isinstance(v, (int, float))]
        if agg_func == "count":
            val = len(vals)
        elif not nums:
            val = 0
        elif agg_func == "sum":
            val = sum(nums)
        elif agg_func == "mean":
            val = sum(nums) / len(nums)
        elif agg_func == "min":
            val = min(nums)
        elif agg_func == "max":
            val = max(nums)
        else:
            val = sum(nums)
        out.append({"group": k, "value": val})
    return out


SYSTEM_PROMPT_PREFIX = """\
You are a data analyst copilot.

- Use the available functions to fetch and aggregate data.
- Remember Monty sandbox restrictions: no imports, no dict mutation, no augmented assignment.
- Return a dict with key `summary` (markdown string). Keep it concise.
"""

agent = CodeModeAgent(
    tools=[fetch_data, group_and_aggregate],
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)


@task_env.task(report=True)
async def analyze(request: str) -> str:
    """Run a durable CodeModeAgent analysis inside a Flyte task."""
    result = await agent.run(request, history=[])
    return result.summary or result.error or ""


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(analyze, request="Show me monthly revenue trends for 2024, broken down by region")
    print(f"Run URL: {run.url}")

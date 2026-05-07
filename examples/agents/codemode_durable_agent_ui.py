"""Durable analytics CodeModeAgent (single file + UI).

This is the chat-UI analogue of `examples/sandbox/codemode/durable_agent.py`,
but kept in a single file: tools, agent, and UI environment.

Key idea: define tools as ``@task_env.task`` so Monty sandbox calls dispatch as
durable Flyte tasks. `CodeModeAgent` introspects ``TaskTemplate.func`` for
prompt signatures + docstrings, so the prompt remains readable.

`AgentChatAppEnvironment(..., passthrough_auth=True)` forwards gateway
credentials so nested Flyte calls (durable tools) can execute.

Run locally::

    uv run python examples/agents/codemode_durable_agent_ui.py
"""

from __future__ import annotations

import json as _json
import pathlib
from typing import Any

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme

task_env = flyte.TaskEnvironment(
    name="codemode-durable-analytics-tools",
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    image=(
        flyte.Image.from_debian_base()
        .with_apt_packages("git")
        .with_pip_packages("httpx", "pydantic-monty", "litellm", "unionai-reuse")
        .with_commands(["uv pip install git+https://www.github.com/flyteorg/flyte-sdk.git@ef1fdf45"])
    ),
    resources=flyte.Resources(cpu=2, memory="1Gi"),
    # reusable=flyte.ReusePolicy(replicas=1, concurrency=10),
)

# ---------------------------------------------------------------------------
# Stub datasets + helpers
# ---------------------------------------------------------------------------

CHART_COLORS = [
    "rgba(230, 152, 18, 0.8)",  # union gold
    "rgba(242, 189, 82, 0.8)",
    "rgba(184, 119, 10, 0.8)",
    "rgba(250, 210, 130, 0.8)",
    "rgba(140, 90, 5, 0.8)",
]
CHART_BORDERS = ["#e69812", "#f2bd52", "#b8770a", "#fad282", "#8c5a05"]


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


# ---------------------------------------------------------------------------
# Durable tool tasks
# ---------------------------------------------------------------------------


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
async def create_chart(chart_type: str, title: str, labels: list, values: list) -> str:
    """Generate a self-contained Chart.js HTML snippet (returned to the chat UI)."""
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
            "plugins": {"title": {"display": True, "text": title}},
        },
    }

    return (
        f'<div style="position:relative;height:350px;margin:20px 0;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f"<script>new Chart(document.getElementById('{canvas_id}'),{_json.dumps(config)});</script>"
    )


@task_env.task
async def calculate_statistics(data: list, column: str) -> dict:
    """Calculate basic descriptive statistics for a numeric column."""
    vals = [row[column] for row in data if isinstance(row, dict) and column in row]
    if not vals:
        return {"count": 0, "mean": 0, "min": 0, "max": 0}
    n = len(vals)
    total = sum(vals)
    return {"count": n, "mean": total / n, "min": min(vals), "max": max(vals)}


@task_env.task
async def filter_data(data: list, column: str, operator: str, value: object) -> list:
    """Filter rows where *column* matches the condition."""
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
    out = []
    for row in data:
        if isinstance(row, dict) and column in row and fn(row[column], value):
            out.append(row)
    return out


@task_env.task
async def group_and_aggregate(data: list, group_by: str, agg_column: str, agg_func: str) -> list:
    """Group rows and aggregate a numeric column."""
    # Monty disallows dict mutation in sandbox code, but this tool runs outside Monty.
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


@task_env.task
async def sort_data(data: list, column: str, descending: bool = False) -> list:
    """Sort rows by a column."""
    rows = [r for r in data if isinstance(r, dict) and column in r]
    return sorted(rows, key=lambda r: r[column], reverse=descending)


SYSTEM_PROMPT_PREFIX = """\
You are a data analyst copilot.

- Use the available functions to fetch, filter, aggregate, and chart data.
- Remember Monty sandbox restrictions: no imports, no dict mutation, no augmented assignment.
- Return a dict with keys: summary (markdown string) and charts (list of HTML snippets from create_chart).
"""

agent = CodeModeAgent(
    tools=[fetch_data, create_chart, calculate_statistics, filter_data, group_and_aggregate, sort_data],
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)


@task_env.task(report=True)
async def task_entrypoint(message: str, history: list[dict[str, str]]) -> dict[str, object]:
    """Entrypoint for the durable CodeModeAgent analysis inside a Flyte task."""
    result = await agent.run(message, history=history)
    return {
        "code": result.code,
        "charts": result.charts,
        "summary": result.summary,
        "error": result.error,
        "attempts": result.attempts,
    }


env = AgentChatAppEnvironment(
    name="codemode-durable-analytics-ui",
    agent=agent,
    task_entrypoint=task_entrypoint,
    title="Durable analytics CodeModeAgent",
    subtitle="LLM-generated Monty code calling durable Flyte task tools.",
    theme=CustomTheme(accent_color="#e69812", accent_hover_color="#f2bd52", button_text_color="#0a0a0f"),
    passthrough_auth=True,
    prompt_nudges=[
        {
            "label": "Show me the data",
            "prompt": "I want to see the data for the sales_2024 dataset.",
        },
        {
            "label": "Monthly revenue trends",
            "prompt": "I want to see the monthly revenue trends for 2024, broken down by region.",
        },
    ],
    depends_on=[task_env],
    image=(
        flyte.Image.from_debian_base()
        .with_pip_packages("litellm", "pydantic-monty==0.0.17", "uvicorn", "fastapi", "flyte[sandbox]")
        .with_local_v2()
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    deployments = flyte.deploy(env)
    print(f"Durable CodeModeAgent UI: {deployments[0].summary_repr()}")

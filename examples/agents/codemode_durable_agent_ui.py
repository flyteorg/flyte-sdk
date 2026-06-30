"""Durable analytics code-mode agent (single file + UI).

This is the chat-UI analogue of `examples/sandbox/codemode/durable_agent.py`,
but kept in a single file: tools, agent, and UI environment.

Key idea: define tools as ``@task_env.task`` so Monty sandbox calls dispatch as
durable Flyte tasks. ``Agent`` in ``code_mode`` introspects ``TaskTemplate.func``
for prompt signatures + docstrings, so the prompt remains readable. The task
entrypoint uses ``await agent.run.aio(...)`` because ``agent.run`` is
synchronous by default; the chat UI calls ``run.aio`` automatically when
routing requests through the parent task.

`AgentChatAppEnvironment(..., passthrough_auth=True)` forwards gateway
credentials so nested Flyte calls (durable tools) can execute.

Structured results (summary + charts) are extracted *at the example level*: the
core ``Agent`` in code mode returns the model's final reply as plain text, so we
ask the model to make that reply a single JSON object and parse it in the task
entrypoint (see ``_extract_structured_result``). Charts are passed back as small
specs and rendered to Chart.js HTML here, avoiding round-tripping large markup
through the LLM.

Run locally::

    uv run python examples/agents/codemode_durable_agent_ui.py
"""

from __future__ import annotations

import json as _json
import pathlib
import re
from typing import Any

import flyte
from flyte.ai.agents import Agent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme

task_env = flyte.TaskEnvironment(
    name="codemode-durable-analytics-tools",
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
    image=(
        flyte.Image.from_debian_base()
        .with_apt_packages("git")
        .with_pip_packages("httpx", "pydantic-monty", "litellm", "unionai-reuse")
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
async def create_chart(chart_type: str, title: str, labels: list, values: list) -> dict:
    """Build a compact Chart.js spec for a chart.

    Returns a small JSON-friendly dict (NOT raw HTML) so the agent can include it
    verbatim in its final structured result without copying large markup through
    the LLM. The chat entrypoint renders each spec to a self-contained Chart.js
    snippet via ``_render_chart_html``.
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
            "plugins": {"title": {"display": True, "text": title}},
        },
    }

    return {"id": canvas_id, "config": config}


@task_env.task
async def calculate_statistics(data: list, column: str) -> dict:
    """Calculate basic descriptive statistics for a numeric column."""
    vals = [row[column] for row in data if isinstance(row, dict) and column in row]
    if not vals:
        return {"count": 0, "mean": 0, "min": 0, "max": 0, "sum": 0, "std": 0}
    n = len(vals)
    total = sum(vals)
    mean = total / n
    # Population standard deviation (no imports; Monty-safe arithmetic).
    var = sum([(v - mean) * (v - mean) for v in vals]) / n
    std = var**0.5
    return {"count": n, "mean": mean, "min": min(vals), "max": max(vals), "sum": total, "std": std}


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
        # Preserve the original column names so downstream code can sort/group
        # using the same keys it requested (e.g. sort by "month" after grouping by "month").
        out.append({group_by: k, agg_column: val})
    return out


@task_env.task
async def sort_data(data: list, column: str, descending: bool = False) -> list:
    """Sort rows by a column."""
    rows = [r for r in data if isinstance(r, dict) and column in r]
    return sorted(rows, key=lambda r: r[column], reverse=descending)


# ---------------------------------------------------------------------------
# Example-level structured-result extraction
#
# Code mode returns the model's final reply as plain text (``AgentResult.summary``).
# Rather than teach the core Agent about charts, this example asks the model to
# make its final reply a single JSON object and parses it here.
# ---------------------------------------------------------------------------


def _render_chart_html(spec: dict[str, Any]) -> str:
    """Render a compact chart spec (from ``create_chart``) to a Chart.js snippet."""
    canvas_id = str(spec.get("id", "chart"))
    config = spec.get("config", {})
    return (
        f'<div style="position:relative;height:350px;margin:20px 0;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f"<script>new Chart(document.getElementById('{canvas_id}'),{_json.dumps(config)});</script>"
    )


def _extract_structured_result(text: str) -> tuple[str, list[str]]:
    """Parse the agent's final reply into ``(summary_markdown, chart_html_snippets)``.

    Expects a JSON object with ``summary`` and ``charts`` keys. ``charts`` entries
    may be compact specs from ``create_chart`` (rendered here) or pre-rendered HTML
    strings. Falls back to treating the whole reply as the summary with no charts.
    """
    if not text:
        return "", []
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
    else:
        # Ignore any prose around the object by grabbing the outermost braces.
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        data = _json.loads(candidate)
    except (ValueError, TypeError):
        return text.strip(), []
    if not isinstance(data, dict):
        return text.strip(), []
    summary = str(data.get("summary") or text.strip())
    charts: list[str] = []
    raw = data.get("charts", [])
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                charts.append(_render_chart_html(entry))
            elif isinstance(entry, str):
                charts.append(entry)
    return summary, charts


SYSTEM_PROMPT_PREFIX = """\
You are a data analyst copilot.

- Use the available functions to fetch, filter, aggregate, and chart data.
- Build charts with create_chart(...); it returns a small chart spec — keep each one.
- Remember Monty sandbox restrictions: no imports, no dict mutation, no augmented assignment.
- When finished, reply with a SINGLE raw JSON object (do NOT wrap it in a code fence) with keys:
  - "summary": a markdown string describing the findings.
  - "charts": a list of the chart spec objects returned by create_chart (use [] if none).
"""

agent = Agent(
    name="durable-analytics-agent",
    instructions=SYSTEM_PROMPT_PREFIX,
    model="claude-haiku-4-5",
    tools=[fetch_data, create_chart, calculate_statistics, filter_data, group_and_aggregate, sort_data],
    code_mode=True,
    max_turns=15,
)


@task_env.task(report=True)
async def codemode_agent_task_entrypoint(message: str, memory: list[dict[str, str]]) -> dict[str, object]:
    """Entrypoint for the durable code-mode agent analysis inside a Flyte task."""
    result = await agent.run.aio(message, memory=memory)
    summary, charts = _extract_structured_result(result.summary)
    return {
        "code": result.code,
        "charts": charts,
        "summary": summary,
        "error": result.error,
        "attempts": result.attempts,
    }


env = AgentChatAppEnvironment(
    name="codemode-durable-analytics-ui",
    agent=agent,
    task_entrypoint=codemode_agent_task_entrypoint,
    title="Durable analytics agent",
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
    print(f"Durable analytics agent UI: {deployments[0].summary_repr()}")

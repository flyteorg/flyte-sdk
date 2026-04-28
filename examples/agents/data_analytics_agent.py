"""Data analytics agent example — summaries and plot plans (stub tools).

Stub tools return a fixed tabular summary, descriptive statistics, and chart
suggestions (no real plotting). Run::

    python examples/agents/data_analytics_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_ROWS = [
    {"region": "North", "sales": 120, "units": 400},
    {"region": "South", "sales": 95, "units": 310},
    {"region": "East", "sales": 140, "units": 450},
    {"region": "West", "sales": 88, "units": 290},
]

_STUB_STATS = {
    "count": 4,
    "sales_mean": 110.75,
    "sales_std": 22.9,
    "sales_min": 88,
    "sales_max": 140,
    "units_sum": 1450,
}


async def load_sample_dataset_stub(dataset_name: str) -> dict[str, object]:
    """Return a small hard-coded table preview (stub)."""
    return {
        "dataset_name": dataset_name or "demo_sales",
        "columns": list(_STUB_ROWS[0].keys()),
        "row_count": len(_STUB_ROWS),
        "sample_rows": _STUB_ROWS,
    }


async def compute_summary_statistics_stub() -> dict[str, object]:
    """Return fixed summary statistics for the stub dataset."""
    return {"statistics": _STUB_STATS}


async def suggest_visualizations_stub() -> list[dict[str, str]]:
    """Return suggested chart types and encodings (stub; no images)."""
    return [
        {
            "chart": "bar",
            "title": "Sales by region",
            "x": "region",
            "y": "sales",
            "note": "Stub: would use matplotlib/plotly in a real pipeline.",
        },
        {
            "chart": "scatter",
            "title": "Sales vs units",
            "x": "sales",
            "y": "units",
            "note": "Stub: check linearity and outliers.",
        },
    ]


async def run_hypothesis_stub(
    hypothesis: str,
) -> dict[str, str]:
    """Return a canned 'analysis' result for demos (stub)."""
    return {
        "hypothesis": hypothesis,
        "result": "not_tested",
        "interpretation": (
            "Stub: In production you would fit a model or run a formal test. "
            "For this demo, assume assumptions hold only for illustration."
        ),
    }


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Final Markdown response for the chat UI."""
    parts = [f"## {title}", "", body]
    if links:
        parts += ["", "**References:**", *[f"- {u}" for u in links]]
    return {"summary": "\n".join(parts)}


ALL_TOOLS: dict[str, Callable] = {
    "load_sample_dataset_stub": load_sample_dataset_stub,
    "compute_summary_statistics_stub": compute_summary_statistics_stub,
    "suggest_visualizations_stub": suggest_visualizations_stub,
    "run_hypothesis_stub": run_hypothesis_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a data analytics copilot. Tools return stub tabular data and chart \
suggestions — describe what plots would show and how to interpret them. \
If the user brings their own numbers, still ground recommendations in tool \
outputs where possible. Always finish with format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="data-analytics-agent-demo",
    agent=agent,
    title="Data analytics agent (demo)",
    subtitle="Summary stats and chart plans from stub data.",
    theme=CustomTheme(accent_color="#0EA5E9", accent_hover_color="#38BDF8", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {"label": "Explore data", "prompt": "Summarize the stub dataset and key takeaways for a stakeholder email."},
        {"label": "Charts", "prompt": "Which visualizations should we build first and why?"},
        {"label": "Hypothesis", "prompt": "We believe Southern regions underperform — what would you check?"},
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages("litellm", "pydantic-monty==0.0.8", "uvicorn", "fastapi", "flyte[sandbox]")
    .with_local_v2(),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    handle = flyte.serve(env)
    print(f"Data analytics agent: {handle.url}")

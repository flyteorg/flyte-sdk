"""Data analytics agent example — EDA, quality, and stakeholder storytelling (stub tools).

Models a typical analytics workflow: clarify objective, profile data, clean
issues, explore distributions, visualize, and communicate — using **fixed**
tables and metrics (no real warehouse or plots). Patterns align with common
EDA guidance: profiling, summaries, visualization choices, and narrative.

Run::

    python examples/agents/data_analytics_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai.agents import CodeModeAgent
from flyte.ai.chat import AgentChatAppEnvironment, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_ROWS = [
    {"region": "North", "sales": 120, "units": 400, "segment": "enterprise"},
    {"region": "South", "sales": 95, "units": 310, "segment": "smb"},
    {"region": "East", "sales": 140, "units": 450, "segment": "enterprise"},
    {"region": "West", "sales": 88, "units": 290, "segment": "smb"},
]

_STUB_STATS = {
    "count": 4,
    "sales_mean": 110.75,
    "sales_std": 22.9,
    "sales_min": 88,
    "sales_max": 140,
    "units_sum": 1450,
}

_STUB_PROFILE = {
    "row_count": 4,
    "columns": {
        "region": {"dtype": "string", "null_pct": 0.0, "distinct": 4},
        "sales": {"dtype": "float", "null_pct": 0.0, "distinct": 4},
        "units": {"dtype": "int", "null_pct": 0.0, "distinct": 4},
        "segment": {"dtype": "string", "null_pct": 0.0, "distinct": 2},
    },
}

_STUB_QUALITY = {
    "duplicate_rows_stub": 0,
    "rows_with_any_null_stub": 0,
    "outlier_flags_stub": [{"region": "West", "column": "sales", "z_score_stub": -1.0}],
    "recommended_fixes_stub": ["Validate West region sales with source system"],
}

_STUB_CORRELATIONS = [
    {"pair": ("sales", "units"), "pearson_r_stub": 0.94, "note": "Stub correlation — illustrative only."},
    {"pair": ("sales", "region_encoded_stub"), "pearson_r_stub": 0.12, "note": "Region encoded ordinally for demo."},
]


async def get_business_objective_templates_stub(
    objective_type: Literal["growth", "efficiency", "risk"],
) -> dict[str, list[str]]:
    """Return questions to clarify the analytics ask (stub)."""
    templates = {
        "growth": [
            "Which revenue lever are we optimizing (volume, price, mix)?",
            "What decision changes if the analysis is positive vs flat?",
            "What is the evaluation window and baseline?",
        ],
        "efficiency": [
            "Which cost or latency KPI owns the outcome?",
            "What constraints (SLA, budget) must the recommendation respect?",
        ],
        "risk": [
            "What false positive vs false negative tradeoff is acceptable?",
            "Which stakeholders sign off on mitigations?",
        ],
    }
    return {"questions": templates[objective_type]}


async def load_sample_dataset_stub(dataset_name: str) -> dict[str, object]:
    """Return a small hard-coded table preview (stub)."""
    return {
        "dataset_name": dataset_name or "demo_sales",
        "columns": list(_STUB_ROWS[0].keys()),
        "row_count": len(_STUB_ROWS),
        "sample_rows": _STUB_ROWS,
    }


async def get_data_profile_stub() -> dict[str, object]:
    """Return column-level dtypes, null rates, and cardinality (stub)."""
    return {"profile": _STUB_PROFILE}


async def get_data_quality_report_stub() -> dict[str, object]:
    """Return duplicate/null/outlier flags and suggested fixes (stub)."""
    return dict(_STUB_QUALITY)


async def compute_summary_statistics_stub() -> dict[str, object]:
    """Return fixed summary statistics for the stub dataset."""
    return {"statistics": _STUB_STATS}


async def get_correlation_insights_stub() -> list[dict[str, object]]:
    """Return stub pairwise correlation summaries."""
    return list(_STUB_CORRELATIONS)


async def segment_comparison_stub(
    segment_col: Literal["segment"],
) -> dict[str, object]:
    """Return stub aggregates by segment value."""
    return {
        "segment_column": segment_col,
        "by_segment_stub": {
            "enterprise": {"mean_sales": 130.0, "n": 2},
            "smb": {"mean_sales": 91.5, "n": 2},
        },
        "note": "Stub aggregates — not computed from live warehouse.",
    }


async def suggest_visualizations_stub() -> list[dict[str, str]]:
    """Return suggested chart types and encodings (stub; no images)."""
    return [
        {
            "chart": "bar",
            "title": "Sales by region",
            "x": "region",
            "y": "sales",
            "note": "Compare magnitudes; sort bars for readability.",
        },
        {
            "chart": "scatter",
            "title": "Sales vs units",
            "x": "sales",
            "y": "units",
            "note": "Check linearity; label outliers (see quality report).",
        },
        {
            "chart": "heatmap",
            "title": "Correlation matrix (illustrative)",
            "x": "numeric_features",
            "y": "numeric_features",
            "note": "Stub: in production build from full feature set.",
        },
    ]


async def get_cohort_retention_stub() -> dict[str, object]:
    """Return fake weekly retention curve for storytelling demos (stub)."""
    return {
        "cohort": "signup_week_stub_12",
        "retention_by_week_stub": [1.0, 0.62, 0.48, 0.41, 0.36, 0.33],
        "note": "Synthetic retention — replace with warehouse cohort query.",
    }


async def get_stakeholder_deck_outline_stub() -> list[dict[str, str]]:
    """Return slide titles for an exec readout (stub)."""
    return [
        {"slide": "Context & decision", "bullets_stub": "What question we answered"},
        {"slide": "Data & quality", "bullets_stub": "Source, freshness, caveats"},
        {"slide": "Key findings", "bullets_stub": "3 bullets max"},
        {"slide": "Recommendations", "bullets_stub": "Owner, metric, timeline"},
        {"slide": "Appendix", "bullets_stub": "Charts & definitions"},
    ]


async def run_hypothesis_stub(hypothesis: str) -> dict[str, str]:
    """Return a canned 'analysis' result for demos (stub)."""
    return {
        "hypothesis": hypothesis,
        "result": "not_tested",
        "interpretation": (
            "Stub: In production you would fit a model or run a formal test. "
            "Use segment_comparison_stub and correlations only as qualitative hints."
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
    "get_business_objective_templates_stub": get_business_objective_templates_stub,
    "load_sample_dataset_stub": load_sample_dataset_stub,
    "get_data_profile_stub": get_data_profile_stub,
    "get_data_quality_report_stub": get_data_quality_report_stub,
    "compute_summary_statistics_stub": compute_summary_statistics_stub,
    "get_correlation_insights_stub": get_correlation_insights_stub,
    "segment_comparison_stub": segment_comparison_stub,
    "suggest_visualizations_stub": suggest_visualizations_stub,
    "get_cohort_retention_stub": get_cohort_retention_stub,
    "get_stakeholder_deck_outline_stub": get_stakeholder_deck_outline_stub,
    "run_hypothesis_stub": run_hypothesis_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a senior data analyst copilot. Tools emit **stub** profiling, quality, \
and visualization ideas — never claim production SQL or p-values were run. \
Start from business questions, then chain profile → quality → stats → plots. \
Close with actionable recommendations and format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-haiku-4-5",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="data-analytics-agent-demo",
    agent=agent,
    title="Data analytics agent (demo)",
    subtitle="EDA, quality checks, and exec narrative — all stub data.",
    theme=CustomTheme(accent_color="#0EA5E9", accent_hover_color="#38BDF8", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {
            "label": "Exec one-pager",
            "prompt": "Using deck outline + summary stats, write an exec summary of the stub regional sales dataset.",
        },
        {
            "label": "EDA checklist",
            "prompt": (
                "Walk through profile, quality, and visualization tools in order and list what you would verify next."
            ),
        },
        {
            "label": "SMB vs enterprise",
            "prompt": "Compare enterprise vs SMB using segment_comparison_stub and suggest two follow-up analyses.",
        },
        {
            "label": "Quality triage",
            "prompt": "Interpret get_data_quality_report_stub and propose remediation steps before modeling.",
        },
        {
            "label": "Hypothesis framing",
            "prompt": "We think West underperforms — frame hypotheses and map each to a tool from this agent.",
        },
        {
            "label": "Retention story",
            "prompt": "Narrate the fake cohort retention curve for a growth review and caveats.",
        },
        {
            "label": "Chart plan",
            "prompt": "Pick two charts from suggest_visualizations_stub and describe axes, audience, and takeaway.",
        },
        {
            "label": "Objective clarity",
            "prompt": (
                "Use growth objective template questions to clarify an ambiguous ask: 'make the dashboard better'."
            ),
        },
        {
            "label": "Correlation caution",
            "prompt": "Explain get_correlation_insights_stub without over-claiming causation.",
        },
        {
            "label": "Handoff to DS",
            "prompt": "Write a short brief for a data scientist including data caveats and suggested model targets.",
        },
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages("litellm", "pydantic-monty==0.0.17", "uvicorn", "fastapi", "flyte[sandbox]")
    .with_local_v2(),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    handle = flyte.serve(env)
    print(f"Data analytics agent: {handle.url}")

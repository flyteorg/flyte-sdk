"""Financial analytics agent example — news, quotes, and plans (stub tools).

All market data and news are hard-coded for demonstration. Not financial advice.
Run::

    python examples/agents/financial_analytics_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_NEWS = [
    {
        "headline": "Central bank signals cautious stance on rates",
        "sentiment": "neutral",
        "impact_sectors": ["financials", "real_estate"],
    },
    {
        "headline": "Large cap tech earnings beat expectations",
        "sentiment": "positive",
        "impact_sectors": ["technology", "communication"],
    },
]

_STUB_QUOTES = {
    "DEMO": {"price": 142.5, "change_pct": 1.2, "currency": "USD"},
    "EXAMPLE": {"price": 38.1, "change_pct": -0.4, "currency": "USD"},
}

_STUB_SECTOR = {
    "technology": {"ytd_return_pct": 12.4, "volatility": "high"},
    "healthcare": {"ytd_return_pct": 4.1, "volatility": "medium"},
    "utilities": {"ytd_return_pct": -1.2, "volatility": "low"},
}


async def get_market_news_stub(
    topic: Literal["macro", "equities", "all"] = "all",
) -> dict[str, object]:
    """Return stub headlines and tags (not live news)."""
    return {"topic": topic, "articles": _STUB_NEWS}


async def get_quote_stub(symbol: str) -> dict[str, object]:
    """Return a fake last price for demo symbols (stub)."""
    key = symbol.upper().strip() or "DEMO"
    q = _STUB_QUOTES.get(key, {"price": 100.0, "change_pct": 0.0, "currency": "USD"})
    return {"symbol": key, **q}


async def get_sector_snapshot_stub(
    sector: Literal["technology", "healthcare", "utilities"],
) -> dict[str, object]:
    """Return stub sector-level metrics."""
    return {sector: _STUB_SECTOR.get(sector, _STUB_SECTOR["technology"])}


async def build_investment_plan_stub(
    risk_profile: Literal["conservative", "balanced", "growth"],
) -> dict[str, object]:
    """Return a template allocation — educational stub only, not advice."""
    templates = {
        "conservative": {"bonds": 60, "equities": 30, "cash": 10},
        "balanced": {"bonds": 40, "equities": 50, "cash": 10},
        "growth": {"bonds": 15, "equities": 80, "cash": 5},
    }
    return {
        "risk_profile": risk_profile,
        "illustrative_allocation_pct": templates[risk_profile],
        "caveats": [
            "Stub demo only — not personalized advice.",
            "Past performance does not guarantee future results.",
            "Consult a licensed professional before acting.",
        ],
    }


async def suggest_charts_finance_stub() -> list[dict[str, str]]:
    """Stub chart suggestions for a pitch deck or memo."""
    return [
        {
            "chart": "line",
            "title": "Illustrative equity path (hypothetical)",
            "note": "Stub: replace with backtested or actual time series.",
        },
        {
            "chart": "stacked_bar",
            "title": "Allocation by sleeve",
            "note": "Stub: tie to build_investment_plan_stub output.",
        },
    ]


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Final Markdown for chat UI; include disclaimers in body when giving plans."""
    parts = [f"## {title}", "", body]
    if links:
        parts += ["", "**References:**", *[f"- {u}" for u in links]]
    return {"summary": "\n".join(parts)}


ALL_TOOLS: dict[str, Callable] = {
    "get_market_news_stub": get_market_news_stub,
    "get_quote_stub": get_quote_stub,
    "get_sector_snapshot_stub": get_sector_snapshot_stub,
    "build_investment_plan_stub": build_investment_plan_stub,
    "suggest_charts_finance_stub": suggest_charts_finance_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a financial analytics assistant for **education and demos only**. \
Tools return stub news and prices — never claim data is live. When discussing \
investments, lead with caveats and use build_investment_plan_stub for \
illustrative allocations only. Always end with format_response(title, body, links) \
and remind users this is not financial advice.
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="financial-analytics-agent-demo",
    agent=agent,
    title="Financial analytics agent (demo)",
    subtitle="Stub news, quotes, and illustrative plans — not advice.",
    theme=CustomTheme(accent_color="#22C55E", accent_hover_color="#4ADE80", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {
            "label": "Morning brief",
            "prompt": "Summarize stub macro and equity headlines for a portfolio manager huddle.",
        },
        {"label": "Quote check", "prompt": "What does the stub data say for symbol DEMO?"},
        {"label": "Plan outline", "prompt": "Outline a balanced-risk illustrative allocation and risks to monitor."},
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
    print(f"Financial analytics agent: {handle.url}")

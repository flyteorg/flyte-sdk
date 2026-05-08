"""Financial analytics agent example — memos, risk, and scenarios (stub tools).

Models investment-committee style workflows: thesis, fundamentals snapshot,
peer context, pre-mortem / bear case, and scenario returns — all **hard-coded**
for education and UI demos. **Not financial advice**; not live market data.

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
    {
        "headline": "Energy complex firms on supply disruption headlines",
        "sentiment": "mixed",
        "impact_sectors": ["energy", "industrials"],
    },
]

_STUB_QUOTES = {
    "DEMO": {"price": 142.5, "change_pct": 1.2, "currency": "USD", "market_cap_bn_stub": 58.2},
    "EXAMPLE": {"price": 38.1, "change_pct": -0.4, "currency": "USD", "market_cap_bn_stub": 4.1},
}

_STUB_SECTOR = {
    "technology": {"ytd_return_pct": 12.4, "volatility": "high"},
    "healthcare": {"ytd_return_pct": 4.1, "volatility": "medium"},
    "utilities": {"ytd_return_pct": -1.2, "volatility": "low"},
}

_STUB_MACRO = {
    "policy_rate_pct_stub": 5.25,
    "inflation_yoy_pct_stub": 2.8,
    "credit_spread_bps_stub": 112,
    "liquidity_note_stub": "Illustrative conditions — not a live macro feed.",
}

_STUB_PEER_COMPS = [
    {"name": "Peer A", "ev_ebitda_stub": 14.2, "revenue_growth_pct_stub": 11},
    {"name": "Peer B", "ev_ebitda_stub": 12.0, "revenue_growth_pct_stub": 8},
    {"name": "DEMO (subject)", "ev_ebitda_stub": 13.1, "revenue_growth_pct_stub": 10},
]

_STUB_MEMO_SECTIONS = [
    "Executive summary & recommendation",
    "Investment thesis (3 falsifiable pillars)",
    "Business overview & moat",
    "Financial history & projections (base / upside / downside)",
    "Peers & valuation context",
    "Risks & mitigations (incl. pre-mortem)",
    "ESG / governance considerations (if material)",
    "Appendix: assumptions, sensitivities, data sources",
]

_STUB_PREMORTEM = [
    "Key customer concentration leads to revenue cliff",
    "Regulatory change invalidates core product",
    "Execution miss on integration; synergy case fails",
    "Multiple compression sector-wide despite intact fundamentals",
]

_STUB_EARNINGS = [
    {
        "symbol": "DEMO",
        "date_stub": "next_thursday",
        "consensus_eps_stub": 1.12,
        "whisper_note_stub": "no live whisper",
    },
    {
        "symbol": "EXAMPLE",
        "date_stub": "in_3_weeks",
        "consensus_eps_stub": 0.41,
        "whisper_note_stub": "no live whisper",
    },
]


async def get_market_news_stub(
    topic: Literal["macro", "equities", "all"] = "all",
) -> dict[str, object]:
    """Return stub headlines and tags (not live news)."""
    return {"topic": topic, "articles": _STUB_NEWS}


async def get_macro_snapshot_stub() -> dict[str, object]:
    """Return illustrative macro indicators (stub)."""
    return dict(_STUB_MACRO)


async def get_quote_stub(symbol: str) -> dict[str, object]:
    """Return a fake last price for demo symbols (stub)."""
    key = symbol.upper().strip() or "DEMO"
    q = _STUB_QUOTES.get(key, {"price": 100.0, "change_pct": 0.0, "currency": "USD", "market_cap_bn_stub": 10.0})
    return {"symbol": key, **q}


async def get_sector_snapshot_stub(
    sector: Literal["technology", "healthcare", "utilities"],
) -> dict[str, object]:
    """Return stub sector-level metrics."""
    return {sector: _STUB_SECTOR.get(sector, _STUB_SECTOR["technology"])}


async def get_peer_comps_table_stub() -> list[dict[str, object]]:
    """Return a canned peer valuation table (stub multiples)."""
    return list(_STUB_PEER_COMPS)


async def get_investment_memo_outline_stub() -> dict[str, object]:
    """Return standard memo section headings (stub template)."""
    return {"sections": list(_STUB_MEMO_SECTIONS)}


async def get_bear_case_pre_mortem_stub() -> dict[str, object]:
    """Return illustrative failure modes for a pre-mortem exercise (stub)."""
    return {"failure_modes_stub": list(_STUB_PREMORTEM), "note": "Educational exercise only."}


async def get_earnings_calendar_stub() -> list[dict[str, object]]:
    """Return fake upcoming earnings rows (stub)."""
    return list(_STUB_EARNINGS)


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


async def get_scenario_returns_stub() -> dict[str, object]:
    """Return hypothetical 12m return bands by scenario (stub)."""
    return {
        "downside_pct_stub": -18,
        "base_pct_stub": 7,
        "upside_pct_stub": 22,
        "note": "Illustrative only — not a forecast or backtest output.",
    }


async def suggest_charts_finance_stub() -> list[dict[str, str]]:
    """Stub chart suggestions for a pitch deck or memo."""
    return [
        {
            "chart": "line",
            "title": "Illustrative revenue & margin trajectory",
            "note": "Stub: replace with audited financials.",
        },
        {
            "chart": "stacked_bar",
            "title": "Allocation by sleeve",
            "note": "Tie to build_investment_plan_stub for portfolio demos.",
        },
        {
            "chart": "waterfall",
            "title": "Bridge from LTM EBITDA to target",
            "note": "Use in thesis section with explicit assumptions.",
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
    "get_macro_snapshot_stub": get_macro_snapshot_stub,
    "get_quote_stub": get_quote_stub,
    "get_sector_snapshot_stub": get_sector_snapshot_stub,
    "get_peer_comps_table_stub": get_peer_comps_table_stub,
    "get_investment_memo_outline_stub": get_investment_memo_outline_stub,
    "get_bear_case_pre_mortem_stub": get_bear_case_pre_mortem_stub,
    "get_earnings_calendar_stub": get_earnings_calendar_stub,
    "build_investment_plan_stub": build_investment_plan_stub,
    "get_scenario_returns_stub": get_scenario_returns_stub,
    "suggest_charts_finance_stub": suggest_charts_finance_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a financial analytics **education** assistant. Tools are stubbed — \
never present prices or macro as live. For equity-style questions, weave \
together memo outline, peers, earnings stub, and pre-mortem. For portfolio \
questions, use build_investment_plan_stub plus scenario bands with heavy \
disclaimers. Always end with format_response(title, body, links) and state \
this is not financial advice.
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-haiku-4-5",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="financial-analytics-agent-demo",
    agent=agent,
    title="Financial analytics agent (demo)",
    subtitle="Memo, risk, and scenario stubs — not advice or live data.",
    theme=CustomTheme(accent_color="#22C55E", accent_hover_color="#4ADE80", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {
            "label": "IC memo skeleton",
            "prompt": "Fill the investment memo outline with stub-aware bullets for a hypothetical DEMO long thesis.",
        },
        {
            "label": "Pre-mortem",
            "prompt": "Run a pre-mortem using bear_case tools and tie mitigations to thesis pillars.",
        },
        {
            "label": "Morning brief",
            "prompt": "Combine macro snapshot and news stub for a fictional PM standup.",
        },
        {
            "label": "Peer valuation",
            "prompt": "Interpret the peer comps table and what additional diligence you would demand.",
        },
        {
            "label": "Earnings prep",
            "prompt": "Draft questions for management ahead of the stub DEMO earnings date.",
        },
        {
            "label": "Sector rotation",
            "prompt": "Compare technology vs utilities sector snapshots and discuss diversification (educational).",
        },
        {
            "label": "Illustrative plan",
            "prompt": "Explain balanced vs growth stub allocations and scenario return bands for a novice.",
        },
        {
            "label": "Risk committee",
            "prompt": "List top risks from pre-mortem plus macro snapshot for a risk committee slide.",
        },
        {
            "label": "Deck charts",
            "prompt": "Pick three finance chart stubs and say what data would replace them in production.",
        },
        {
            "label": "Quote drill-down",
            "prompt": "Summarize stub DEMO quote fields and what is still unknown for a full equity write-up.",
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
    print(f"Financial analytics agent: {handle.url}")

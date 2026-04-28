"""Logistics / supply chain agent example — inventory and reports (stub tools).

Hard-coded industry snapshots for decision-support demos. Run::

    python examples/agents/logistics_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_INVENTORY = [
    {"sku": "SKU-1001", "on_hand": 2400, "reorder_point": 800, "days_cover": 18},
    {"sku": "SKU-2044", "on_hand": 120, "reorder_point": 200, "days_cover": 5},
    {"sku": "SKU-3300", "on_hand": 50, "reorder_point": 150, "days_cover": 2},
]

_STUB_FORECAST = {
    "horizon_weeks": 8,
    "demand_index": [1.0, 1.02, 1.05, 1.08, 1.06, 1.04, 1.03, 1.01],
    "confidence_band_stub": "±6% around baseline (illustrative)",
}

_STUB_RISKS = [
    {"id": "R1", "title": "Port congestion — inbound Asia lanes", "severity": "high"},
    {"id": "R2", "title": "Carrier capacity tightens in Q3", "severity": "medium"},
    {"id": "R3", "title": "Single-source packaging supplier", "severity": "medium"},
]


async def get_inventory_snapshot_stub(
    warehouse: str,
) -> dict[str, object]:
    """Return fixed SKU lines for a warehouse (stub)."""
    return {"warehouse": warehouse or "WH-EAST-01", "lines": _STUB_INVENTORY}


async def get_demand_forecast_stub(
    product_line: Literal["consumer", "industrial", "all"],
) -> dict[str, object]:
    """Return stub demand curve metadata."""
    return {"product_line": product_line, "forecast": _STUB_FORECAST}


async def identify_supply_chain_risks_stub() -> list[dict[str, str]]:
    """Return canned risk register entries."""
    return list(_STUB_RISKS)


async def recommend_actions_stub(
    priority: Literal["cost", "service_level", "resilience"],
) -> list[dict[str, str]]:
    """Return illustrative action list tuned to a priority (stub)."""
    actions = {
        "cost": [
            {"action": "Renegotiate secondary carrier rates", "impact": "medium", "lead_time_weeks": "4"},
            {"action": "Consolidate LTL into milk-runs", "impact": "high", "lead_time_weeks": "8"},
        ],
        "service_level": [
            {"action": "Raise safety stock on SKU-3300", "impact": "high", "lead_time_weeks": "1"},
            {"action": "Split inbound across two ports", "impact": "medium", "lead_time_weeks": "6"},
        ],
        "resilience": [
            {"action": "Qualify second packaging vendor", "impact": "high", "lead_time_weeks": "12"},
            {"action": "Regional buffer stock pilot", "impact": "medium", "lead_time_weeks": "10"},
        ],
    }
    return actions[priority]


async def get_sla_metrics_stub() -> dict[str, float | str]:
    """Return stub OTIF and fill-rate style KPIs."""
    return {
        "otif_pct": 94.2,
        "fill_rate_pct": 97.5,
        "avg_backorder_lines": 3.1,
        "period": "last_30_days_stub",
    }


async def format_response(
    title: str,
    body: str,
    links: list[str] | None = None,
) -> FormattedResponse:
    """Final Markdown for the chat UI."""
    parts = [f"## {title}", "", body]
    if links:
        parts += ["", "**References:**", *[f"- {u}" for u in links]]
    return {"summary": "\n".join(parts)}


ALL_TOOLS: dict[str, Callable] = {
    "get_inventory_snapshot_stub": get_inventory_snapshot_stub,
    "get_demand_forecast_stub": get_demand_forecast_stub,
    "identify_supply_chain_risks_stub": identify_supply_chain_risks_stub,
    "recommend_actions_stub": recommend_actions_stub,
    "get_sla_metrics_stub": get_sla_metrics_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a logistics and supply chain analyst copilot. All numbers and risks \
from tools are **stubbed** for product demos — frame outputs as illustrations, \
not live ERP data. Recommend concrete next steps and end every answer with \
format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="logistics-agent-demo",
    agent=agent,
    title="Logistics agent (demo)",
    subtitle="Inventory, demand, and risk stubs for decision memos.",
    theme=CustomTheme(accent_color="#64748B", accent_hover_color="#94A3B8", button_text_color="#f8fafc"),
    prompt_nudges=[
        {"label": "Stock alert", "prompt": "Which SKUs need attention and what should ops do this week?"},
        {"label": "Executive summary", "prompt": "One-page brief: demand outlook, risks, and resilience actions."},
        {"label": "Cost push", "prompt": "Prioritize cost-saving actions using the stub forecast and inventory."},
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
    print(f"Logistics agent: {handle.url}")

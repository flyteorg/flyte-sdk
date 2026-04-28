"""Logistics / supply chain agent example — S&OP, KPIs, and resilience (stub tools).

Aligns with common planning themes: demand vs supply, inventory health,
forecast accuracy, perfect order / OTIF components, and executive S&OP-style
briefs — using **fixed** numbers (no ERP connection).

Run::

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
    {"sku": "SKU-1001", "on_hand": 2400, "reorder_point": 800, "days_cover": 18, "abc_class_stub": "A"},
    {"sku": "SKU-2044", "on_hand": 120, "reorder_point": 200, "days_cover": 5, "abc_class_stub": "B"},
    {"sku": "SKU-3300", "on_hand": 50, "reorder_point": 150, "days_cover": 2, "abc_class_stub": "A"},
    {"sku": "SKU-4410", "on_hand": 900, "reorder_point": 400, "days_cover": 11, "abc_class_stub": "C"},
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
    {"id": "R4", "title": "Tariff exposure on 12% of COGS (stub)", "severity": "medium"},
]

_STUB_SOP_STEPS = [
    {"step": 1, "name": "Data gathering & portfolio review", "outputs_stub": "SKU-level demand history"},
    {"step": 2, "name": "Demand review", "outputs_stub": "Consensus forecast + changes"},
    {"step": 3, "name": "Supply review", "outputs_stub": "Capacity, lead time, constraints"},
    {"step": 4, "name": "Pre-S&OP reconciliation", "outputs_stub": "Finance gap vs plan"},
    {"step": 5, "name": "Executive S&OP", "outputs_stub": "Single authorized plan + risks"},
]

_STUB_FORECAST_ACCURACY = {
    "wmape_pct_stub": 11.2,
    "bias_pct_stub": 2.1,
    "horizon_stub": "rolling_12_weeks",
    "note": "Illustrative accuracy — replace with demand planning system metrics.",
}

_STUB_PERFECT_ORDER = {
    "on_time_pct_stub": 96.1,
    "in_full_pct_stub": 97.8,
    "damage_free_pct_stub": 99.0,
    "invoice_accurate_pct_stub": 98.5,
    "composite_perfect_order_pct_stub": 92.4,
}

_STUB_CARRIERS = [
    {"carrier": "BlueFreight", "otif_pct_stub": 95.2, "cost_index_stub": 1.0},
    {"carrier": "RedHaul", "otif_pct_stub": 97.1, "cost_index_stub": 1.08},
]

_STUB_CAPACITY = {
    "warehouse": "WH-EAST-01",
    "utilization_pct_stub": 86,
    "dock_doors_available_stub": 2,
    "labor_hours_gap_stub": -120,
}

_STUB_TARIFF = {
    "exposed_cogs_pct_stub": 12,
    "affected_lanes_stub": ["Asia → US West", "EU → US East"],
    "mitigation_ideas_stub": ["nearshoring pilot", "tariff engineering review", "supplier re-quote"],
}

_STUB_CASH = {
    "cash_to_cash_days_stub": 61,
    "benchmark_peer_days_stub": 48,
    "note": "Stub working-capital snapshot.",
}

_STUB_COLLAB_PROGRAMS = [
    {"program": "VMI pilot (stub)", "coverage_skus_stub": 120, "status": "evaluation"},
    {"program": "Consignment packaging (stub)", "coverage_skus_stub": 45, "status": "live"},
]


async def get_sop_cycle_steps_stub() -> list[dict[str, object]]:
    """Return a five-step S&OP style process (stub)."""
    return list(_STUB_SOP_STEPS)


async def get_inventory_snapshot_stub(warehouse: str) -> dict[str, object]:
    """Return fixed SKU lines for a warehouse (stub)."""
    return {"warehouse": warehouse or "WH-EAST-01", "lines": _STUB_INVENTORY}


async def get_inventory_days_on_hand_stub() -> dict[str, object]:
    """Return DOH style rollups by ABC class (stub)."""
    return {
        "by_class_stub": {"A": {"median_doh": 9}, "B": {"median_doh": 14}, "C": {"median_doh": 22}},
        "note": "Synthetic rollups for demos.",
    }


async def get_demand_forecast_stub(
    product_line: Literal["consumer", "industrial", "all"],
) -> dict[str, object]:
    """Return stub demand curve metadata."""
    return {"product_line": product_line, "forecast": _STUB_FORECAST}


async def get_forecast_accuracy_stub() -> dict[str, object]:
    """Return stub WMAPE / bias style metrics."""
    return dict(_STUB_FORECAST_ACCURACY)


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
            {"action": "Rightsize safety stock on C-class SKUs", "impact": "medium", "lead_time_weeks": "3"},
        ],
        "service_level": [
            {"action": "Raise safety stock on SKU-3300", "impact": "high", "lead_time_weeks": "1"},
            {"action": "Split inbound across two ports", "impact": "medium", "lead_time_weeks": "6"},
            {"action": "Add weekend receiving shift (pilot)", "impact": "medium", "lead_time_weeks": "4"},
        ],
        "resilience": [
            {"action": "Qualify second packaging vendor", "impact": "high", "lead_time_weeks": "12"},
            {"action": "Regional buffer stock pilot", "impact": "medium", "lead_time_weeks": "10"},
            {"action": "Dual-source critical ASIC components", "impact": "high", "lead_time_weeks": "20"},
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


async def get_perfect_order_breakdown_stub() -> dict[str, object]:
    """Return components toward a perfect-order style KPI (stub)."""
    return dict(_STUB_PERFECT_ORDER)


async def get_carrier_scorecard_stub() -> list[dict[str, object]]:
    """Return stub OTIF vs cost index by carrier."""
    return list(_STUB_CARRIERS)


async def get_warehouse_capacity_stub() -> dict[str, object]:
    """Return utilization and labor gap snapshot (stub)."""
    return dict(_STUB_CAPACITY)


async def get_tariff_exposure_stub() -> dict[str, object]:
    """Return illustrative tariff / lane exposure (stub)."""
    return dict(_STUB_TARIFF)


async def get_cash_to_cash_snapshot_stub() -> dict[str, object]:
    """Return working-capital style snapshot (stub)."""
    return dict(_STUB_CASH)


async def get_collaborative_programs_stub() -> list[dict[str, object]]:
    """Return VMI / consignment style program roster (stub)."""
    return list(_STUB_COLLAB_PROGRAMS)


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
    "get_sop_cycle_steps_stub": get_sop_cycle_steps_stub,
    "get_inventory_snapshot_stub": get_inventory_snapshot_stub,
    "get_inventory_days_on_hand_stub": get_inventory_days_on_hand_stub,
    "get_demand_forecast_stub": get_demand_forecast_stub,
    "get_forecast_accuracy_stub": get_forecast_accuracy_stub,
    "identify_supply_chain_risks_stub": identify_supply_chain_risks_stub,
    "recommend_actions_stub": recommend_actions_stub,
    "get_sla_metrics_stub": get_sla_metrics_stub,
    "get_perfect_order_breakdown_stub": get_perfect_order_breakdown_stub,
    "get_carrier_scorecard_stub": get_carrier_scorecard_stub,
    "get_warehouse_capacity_stub": get_warehouse_capacity_stub,
    "get_tariff_exposure_stub": get_tariff_exposure_stub,
    "get_cash_to_cash_snapshot_stub": get_cash_to_cash_snapshot_stub,
    "get_collaborative_programs_stub": get_collaborative_programs_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a supply chain planning and logistics analyst copilot. All KPIs, \
forecasts, and risks are **stubbed** for demos — never imply live ERP or TMS \
feeds. Structure answers like S&OP: demand, supply, inventory, gaps, actions. \
Reference collaborative programs when discussing resilience. Always end with \
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
    subtitle="S&OP, inventory, forecast accuracy, and carrier KPIs — stub data.",
    theme=CustomTheme(accent_color="#64748B", accent_hover_color="#94A3B8", button_text_color="#f8fafc"),
    prompt_nudges=[
        {
            "label": "S&OP exec one-pager",
            "prompt": "Run through S&OP steps with forecast, inventory, and risks for execs.",
        },
        {
            "label": "Stockout prevention",
            "prompt": "Which SKUs are critical and what actions from recommend_actions_stub apply?",
        },
        {
            "label": "Forecast improvement",
            "prompt": "Interpret forecast accuracy stub and propose a 30-day planning cadence.",
        },
        {
            "label": "Perfect order drill-down",
            "prompt": "Use perfect order breakdown to explain weakest component and fixes.",
        },
        {
            "label": "Carrier decision",
            "prompt": "Compare carrier scorecard for cost vs service; recommend a policy (stub).",
        },
        {
            "label": "Warehouse crunch",
            "prompt": "Capacity stub says high utilization — what mitigations align with resilience priority?",
        },
        {"label": "Tariff scenario", "prompt": "Explain tariff exposure stub and tie mitigations to risk register."},
        {"label": "Cash pressure", "prompt": "Relate cash-to-cash snapshot to inventory days and forecast bias."},
        {
            "label": "VMI / consignment",
            "prompt": "How would collaborative programs stub change inventory policy for B-class SKUs?",
        },
        {
            "label": "90-day bridge",
            "prompt": "Build a 90-day demand-supply bridge narrative using demand_index and inventory snapshot.",
        },
    ],
    image=flyte.Image.from_debian_base(install_flyte=False)
    .with_pip_packages("litellm", "pydantic-monty==0.0.8", "uvicorn", "fastapi", "flyte[sandbox]")
    .with_local_v2(),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=flyte.Secret("internal-anthropic-api-key", as_env_var="ANTHROPIC_API_KEY"),
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    handle = flyte.serve(env)
    print(f"Logistics agent: {handle.url}")

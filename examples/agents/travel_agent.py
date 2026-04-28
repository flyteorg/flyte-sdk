"""Travel planning agent example — itineraries from stub data.

Tools return hard-coded destination ideas, seasonal notes, and budget bands.
Run::

    python examples/agents/travel_agent.py
"""

from __future__ import annotations

import pathlib
from typing import Callable, Literal, TypedDict

import flyte
from flyte.ai import AgentChatAppEnvironment, CodeModeAgent, CustomTheme


class FormattedResponse(TypedDict):
    summary: str


_STUB_DESTINATIONS = {
    "city_break": [
        {"name": "Lisbon", "country": "Portugal", "vibe": "hills, tiles, seafood", "budget_band": "mid"},
        {"name": "Montreal", "country": "Canada", "vibe": "food, festivals, bilingual", "budget_band": "mid"},
    ],
    "nature": [
        {"name": "Banff", "country": "Canada", "vibe": "mountains, lakes", "budget_band": "high"},
        {"name": "Costa Rica (Guanacaste)", "country": "Costa Rica", "vibe": "wildlife, beaches", "budget_band": "mid"},
    ],
    "beach": [
        {"name": "Algarve", "country": "Portugal", "vibe": "cliffs, surf", "budget_band": "mid"},
        {"name": "Okinawa", "country": "Japan", "vibe": "islands, snorkeling", "budget_band": "high"},
    ],
}

_STUB_SEASON = {
    "spring": {"crowds": "moderate", "weather": "mild", "tip": "Book popular museums mid-week."},
    "summer": {"crowds": "high", "weather": "warm", "tip": "Fly mid-week; expect peak hotel rates."},
    "fall": {"crowds": "low_moderate", "weather": "pleasant", "tip": "Great shoulder-season deals in Europe."},
    "winter": {"crowds": "varies", "weather": "cold to mild by latitude", "tip": "Check holiday blackout dates."},
}


async def search_destinations_stub(
    trip_style: Literal["city_break", "nature", "beach"],
) -> dict[str, object]:
    """Return canned destination shortlist for a trip style (stub)."""
    return {"trip_style": trip_style, "options": _STUB_DESTINATIONS.get(trip_style, _STUB_DESTINATIONS["city_break"])}


async def get_season_travel_notes_stub(
    season: Literal["spring", "summer", "fall", "winter"],
) -> dict[str, object]:
    """Return generic seasonal travel notes (stub)."""
    return {season: _STUB_SEASON.get(season, _STUB_SEASON["spring"])}


async def estimate_budget_band_stub(
    nights: int,
    travelers: int,
    comfort: Literal["budget", "mid", "luxury"],
) -> dict[str, object]:
    """Return illustrative daily cost ranges in USD (stub, not live pricing)."""
    daily = {"budget": (80, 140), "mid": (180, 320), "luxury": (450, 900)}[comfort]
    return {
        "nights": nights,
        "travelers": travelers,
        "comfort": comfort,
        "illustrative_daily_usd_per_person": {"low": daily[0], "high": daily[1]},
        "note": "Stub ranges for demos — verify with current fares and hotels.",
    }


async def get_visa_health_reminders_stub(destination_region: str) -> dict[str, list[str]]:
    """Return generic reminders (stub; not authoritative)."""
    return {
        "destination_region": destination_region or "general",
        "reminders": [
            "Verify passport validity (often 6+ months remaining).",
            "Check entry rules and visa requirements on official government sites.",
            "Review CDC/WHO guidance for vaccinations relevant to your itinerary.",
        ],
    }


async def build_itinerary_skeleton_stub(
    city: str,
    days: int,
) -> dict[str, object]:
    """Return a day-by-day skeleton itinerary (stub)."""
    days = max(1, min(days, 5))
    outline = []
    for d in range(1, days + 1):
        outline.append(
            {
                "day": d,
                "morning": f"Explore core neighborhood — {city} (stub)",
                "afternoon": "Museum or walking tour",
                "evening": "Local dinner reservation",
            }
        )
    return {"city": city, "days": days, "outline": outline}


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
    "search_destinations_stub": search_destinations_stub,
    "get_season_travel_notes_stub": get_season_travel_notes_stub,
    "estimate_budget_band_stub": estimate_budget_band_stub,
    "get_visa_health_reminders_stub": get_visa_health_reminders_stub,
    "build_itinerary_skeleton_stub": build_itinerary_skeleton_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a travel planning assistant. All destination, pricing, and logistics \
data from tools is **stubbed for demos** — tell the user to verify bookings, \
visas, and health requirements independently. Build practical itineraries and \
always end with format_response(title, body, links).
"""

agent = CodeModeAgent(
    tools=ALL_TOOLS,
    model="claude-sonnet-4-6",
    max_retries=2,
    system_prompt_prefix=SYSTEM_PROMPT_PREFIX,
)

env = AgentChatAppEnvironment(
    name="travel-agent-demo",
    agent=agent,
    title="Travel agent (demo)",
    subtitle="Itineraries and budget bands from stub data.",
    theme=CustomTheme(accent_color="#F97316", accent_hover_color="#FB923C", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {"label": "Weekend city", "prompt": "Plan a 3-day mid-budget city break for two in spring."},
        {"label": "Nature trip", "prompt": "Suggest a 4-day nature-focused trip with a rough budget band."},
        {"label": "Beach", "prompt": "Compare two stub beach destinations for a family in summer."},
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
    print(f"Travel agent: {handle.url}")

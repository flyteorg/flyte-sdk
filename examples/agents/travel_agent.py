"""Travel planning agent example — itineraries, logistics, and constraints (stub tools).

Covers multi-city routing ideas, budget bands, accessibility and family notes,
and responsible-travel hints using **fixed** data (no bookings or live prices).
Patterns reflect common AI trip-planner concerns: timing, transfers, and
shareable day-by-day plans.

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
        {"name": "Copenhagen", "country": "Denmark", "vibe": "design, cycling", "budget_band": "high"},
    ],
    "nature": [
        {"name": "Banff", "country": "Canada", "vibe": "mountains, lakes", "budget_band": "high"},
        {"name": "Costa Rica (Guanacaste)", "country": "Costa Rica", "vibe": "wildlife, beaches", "budget_band": "mid"},
        {
            "name": "Patagonia (El Calafate)",
            "country": "Argentina",
            "vibe": "glaciers, trekking",
            "budget_band": "high",
        },
    ],
    "beach": [
        {"name": "Algarve", "country": "Portugal", "vibe": "cliffs, surf", "budget_band": "mid"},
        {"name": "Okinawa", "country": "Japan", "vibe": "islands, snorkeling", "budget_band": "high"},
        {"name": "Zanzibar", "country": "Tanzania", "vibe": "spice routes, reefs", "budget_band": "mid"},
    ],
}

_STUB_SEASON = {
    "spring": {"crowds": "moderate", "weather": "mild", "tip": "Book popular museums mid-week."},
    "summer": {"crowds": "high", "weather": "warm", "tip": "Fly mid-week; expect peak hotel rates."},
    "fall": {"crowds": "low_moderate", "weather": "pleasant", "tip": "Great shoulder-season deals in Europe."},
    "winter": {"crowds": "varies", "weather": "cold to mild by latitude", "tip": "Check holiday blackout dates."},
}

_STUB_MULTI_CITY = {
    "route_name_stub": "Lisbon → Porto → Madrid",
    "legs_stub": [
        {"from": "Lisbon", "to": "Porto", "mode": "train", "duration_h_stub": 3.2},
        {
            "from": "Porto",
            "to": "Madrid",
            "mode": "flight",
            "duration_h_stub": 4.5,
            "buffer_note": "Allow 2h airport buffer",
        },
    ],
    "pace_note": "Stub: alternate heavy travel days with local rest days.",
}

_STUB_TRANSPORT = [
    {"mode": "metro_day_pass", "when": "dense city cores", "accessibility_stub": "check elevator outages"},
    {"mode": "rideshare", "when": "late night or bulky luggage", "accessibility_stub": "request WAV where available"},
    {"mode": "bike_share", "when": "short hops, good weather", "accessibility_stub": "not all docks step-free"},
]

_STUB_FAMILY = [
    {"activity": "Science museum (half day)", "age_fit_stub": "6+", "tip": "Buy timed tickets morning-of"},
    {"activity": "River walk + playground", "age_fit_stub": "3+", "tip": "Pack snacks; stroller-friendly path stub"},
]

_STUB_SUSTAINABILITY = [
    "Prefer trains under ~4h vs short hops by air when feasible",
    "Choose walkable neighborhoods to reduce daily taxi use",
    "Offset programs: verify additionality and project type (stub list only)",
    "Support local guides and small businesses in peak destinations",
]


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
    days = max(1, min(days, 7))
    outline = []
    for d in range(1, days + 1):
        outline.append(
            {
                "day": d,
                "morning": f"Neighborhood orientation — {city} (stub)",
                "afternoon": "Museum, market, or light activity",
                "evening": "Local dinner; avoid back-to-back late nights",
            }
        )
    return {"city": city, "days": days, "outline": outline}


async def get_multi_city_route_stub() -> dict[str, object]:
    """Return a sample multi-city leg plan with transfer buffers (stub)."""
    return dict(_STUB_MULTI_CITY)


async def get_transport_options_stub() -> list[dict[str, str]]:
    """Return generic modality notes including accessibility prompts (stub)."""
    return list(_STUB_TRANSPORT)


async def get_family_friendly_activities_stub() -> list[dict[str, str]]:
    """Return canned family activities (stub)."""
    return list(_STUB_FAMILY)


async def get_sustainability_tips_stub() -> list[str]:
    """Return responsible-travel suggestion bullets (stub)."""
    return list(_STUB_SUSTAINABILITY)


async def get_dietary_and_accessibility_preferences_stub(
    prefs: str,
) -> dict[str, list[str]]:
    """Echo planning dimensions to consider (stub; prefs string is not parsed)."""
    return {
        "user_note": prefs or "(none provided)",
        "checklist_stub": [
            "Step-free routes and hotel room features",
            "Restaurant allergen cards in local language",
            "Quiet hours for sensory-sensitive travelers",
            "Medical facilities near lodging",
        ],
    }


async def get_local_events_stub(city: str) -> list[dict[str, str]]:
    """Return fake weekend events for narrative flavor (stub)."""
    return [
        {"city": city or "Lisbon", "event": "Street festival (stub)", "day": "Saturday"},
        {"city": city or "Lisbon", "event": "Jazz in the park (stub)", "day": "Sunday evening"},
    ]


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
    "get_multi_city_route_stub": get_multi_city_route_stub,
    "get_transport_options_stub": get_transport_options_stub,
    "get_family_friendly_activities_stub": get_family_friendly_activities_stub,
    "get_sustainability_tips_stub": get_sustainability_tips_stub,
    "get_dietary_and_accessibility_preferences_stub": get_dietary_and_accessibility_preferences_stub,
    "get_local_events_stub": get_local_events_stub,
    "format_response": format_response,
}

SYSTEM_PROMPT_PREFIX = """\
You are a travel planner. All tools are **stubbed** — never imply confirmed \
bookings, live prices, or visa approval. Prefer geographic clustering, \
reasonable pacing, and explicit buffers for multi-city legs. Mention \
sustainability and accessibility when relevant. End with \
format_response(title, body, links).
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
    subtitle="Itineraries, multi-city legs, and family/sustainability stubs.",
    theme=CustomTheme(accent_color="#F97316", accent_hover_color="#FB923C", button_text_color="#0a0a0f"),
    prompt_nudges=[
        {"label": "Weekend city", "prompt": "Plan a 3-day mid-budget city break for two in spring with daily pacing."},
        {
            "label": "Multi-city Europe",
            "prompt": "Use multi-city route stub plus itinerary skeleton for Lisbon as first stop.",
        },
        {
            "label": "Nature + rest",
            "prompt": "Suggest a 5-day nature trip with one explicit rest day using stub destinations.",
        },
        {"label": "Family trip", "prompt": "Build a family-friendly weekend using family activities and budget bands."},
        {
            "label": "Accessible trip",
            "prompt": "Layer accessibility checklist and transport options onto a 4-day city plan.",
        },
        {
            "label": "Sustainable options",
            "prompt": "Propose a trip narrative that applies sustainability tips to a beach style trip.",
        },
        {
            "label": "Remote work month",
            "prompt": "Outline a 14-day slow travel plan mixing work blocks and exploration (stub data).",
        },
        {
            "label": "Conference add-on",
            "prompt": "Assume a conference Mon-Wed in Madrid; add pre-weekend exploration using stubs.",
        },
        {
            "label": "Budget backpacker",
            "prompt": "Maximize budget_band comfort=budget for 10 nights — where would you cut vs splurge?",
        },
        {"label": "Events layer", "prompt": "Weave local_events_stub into a Saturday-heavy itinerary with caveats."},
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
    print(f"Travel agent: {handle.url}")

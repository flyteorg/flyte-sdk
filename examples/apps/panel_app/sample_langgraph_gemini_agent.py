import os
from datetime import datetime, timezone
from typing import Any
from typing import Annotated, Sequence, TypedDict

import flyte
import requests
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

env = flyte.TaskEnvironment(
    name="langgraph-gemini-agent",
    secrets=[flyte.Secret(key="GOOGLE_GEMINI_API_KEY")],
)


class AgentState(TypedDict):
    """The state shared across LangGraph nodes."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


def _geocode_location(location: str) -> tuple[float, float] | None:
    geocode_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geocode_response.raise_for_status()
    geocode_data = geocode_response.json()
    results: list[dict[str, Any]] = geocode_data.get("results", [])
    if not results:
        return None
    return float(results[0]["latitude"]), float(results[0]["longitude"])


@tool("get_weather_forecast")
@flyte.trace
async def get_weather_forecast(location: str, date: str) -> str:
    """Get hourly temperature forecast for a city on a yyyy-mm-dd date."""
    try:
        coordinates = _geocode_location(location)
    except requests.RequestException as e:
        return f"Failed to geocode location due to a network error: {e}"

    if coordinates is None:
        return f"Location not found: {location}"

    latitude, longitude = coordinates
    try:
        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "hourly": "temperature_2m",
                "start_date": date,
                "end_date": date,
                "timezone": "UTC",
            },
            timeout=10,
        )
        forecast_response.raise_for_status()
    except requests.RequestException as e:
        return f"Failed to fetch forecast due to a network error: {e}"

    forecast_data = forecast_response.json().get("hourly", {})
    times: list[str] = forecast_data.get("time", [])
    temperatures: list[float] = forecast_data.get("temperature_2m", [])
    if not times or not temperatures:
        return f"No hourly forecast data returned for {location} on {date}."

    hourly_lines = [
        f"{ts}: {temp}C" for ts, temp in zip(times, temperatures, strict=False)
    ]
    return f"Weather forecast for {location} on {date} (UTC):\n" + "\n".join(
        hourly_lines
    )


TOOLS = [get_weather_forecast]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def _create_model() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_GEMINI_API_KEY is not set. Set it in your environment before running this example."
        )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_retries=2,
        google_api_key=api_key,
    )
    return llm.bind_tools(TOOLS)


def call_model(state: AgentState, config: RunnableConfig) -> dict:
    model = _create_model()
    response = model.invoke(state["messages"], config=config)
    return {"messages": [response]}


async def call_tool(state: AgentState) -> dict:
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = await TOOLS_BY_NAME[tool_call["name"]].ainvoke(
            tool_call["args"]
        )
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    if not state["messages"][-1].tool_calls:
        return "end"
    return "continue"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    workflow.set_entry_point("llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "llm")
    return workflow.compile()


@env.task
async def main(prompt: str) -> str:
    graph = build_graph()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = await graph.ainvoke(
        {
            "messages": [
                (
                    "user",
                    (
                        f"{prompt}\n"
                        "Use the get_weather_forecast tool whenever weather data is "
                        "needed. The tool requires `location` and `date` "
                        f"(yyyy-mm-dd). Use {today} when the user does not provide a date."
                    ),
                )
            ]
        }
    )
    context = result["messages"][-1].content
    if isinstance(context, str):
        return context
    try:
        return context[0]["text"]
    except (IndexError, KeyError):
        return str(context)

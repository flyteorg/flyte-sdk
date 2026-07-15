"""Build your own LangGraph agent — durable tools on Flyte.

This example shows the "bring your own agent SDK" path: you write your own
LangGraph ``StateGraph`` with custom nodes, conditional edges, and tool bindings,
and Flyte provides the durable runtime underneath.

- ``FlyteAgent.durable_tools()`` wraps your Flyte tasks so each tool call
  executes as a durable child action (own container/GPU, retries, caching).
- ``FlyteAgent.build()`` returns a compiled LangGraph ``StateGraph`` you can
  customize further (subgraphs, checkpointing, human-in-the-loop, etc.).
- The graph runs durably inside a Flyte task — every tool call is a
  first-class Flyte node.

Run:  flyte run langgraph_custom_agent.py city_agent --city "Tokyo"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.langgraph import FlyteAgent

env = flyte.TaskEnvironment(
    "langgraph-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="langgraph-custom-agent")
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-core",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-langgraph",
                pre=True,
            ),
        )
    ),
)


# ── Step 1: Define your durable Flyte tasks ──────────────────────────────────
# These are normal Flyte tasks — they can be cached, retried, sized independently.


@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


# ── Step 2: Build your own agent with FlyteAgent ─────────────────────────────
# You control the graph definition — nodes, edges, state, etc.
# FlyteAgent gives you durable tools and a clean builder.


agent = FlyteAgent(
    name="city-facts-graph",
    instructions=(
        "You are a helpful city-facts assistant. Use the available tools to answer "
        "questions about cities. Be concise and accurate."
    ),
)

# Wrap your Flyte tasks as durable LangGraph tool nodes
tools = agent.durable_tools(get_weather, get_population)

# Build the graph — this returns a compiled StateGraph you can customize further
built_graph = agent.build(tools=tools)


# ── Step 3: Run your graph durably inside a Flyte task ───────────────────────
# The graph definition is fully yours; the task is the durable runtime.


@env.task(report=True, retries=3)
async def city_agent(city: str) -> str:
    """Run the custom-built graph durably on Flyte."""
    from langchain_core.messages import HumanMessage

    result = await built_graph.ainvoke(
        {"messages": [HumanMessage(content=f"What's the weather in {city}?")]},
    )
    # Extract the last message content from the result
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
    return str(result) if result else ""


# ── Alternative: Use FlyteAgent.run() for a quick build-and-run ──────────────
# If you don't need to separate graph definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call."""
    quick_agent = FlyteAgent(name="quick-agent")
    durable_tools = quick_agent.durable_tools(get_weather, get_population)
    return await quick_agent.run(
        f"What's the weather in {city}?",
        tools=durable_tools,
    )


# ── Alternative: Build and customize the graph further ───────────────────────
# You can access the compiled graph and add custom nodes or edges.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the graph before running."""
    from langchain_core.messages import HumanMessage

    agent = FlyteAgent(
        name="custom-graph",
        instructions="You are a weather expert. Always include temperature and conditions.",
    )
    built = agent.build(tools=agent.durable_tools(get_weather))

    # You can access and modify the underlying graph here if needed
    # For example, add checkpointing, subgraphs, or human-in-the-loop nodes

    result = await built.ainvoke(
        {"messages": [HumanMessage(content=f"What's the weather in {city}?")]},
    )
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last = messages[-1]
            return last.content if hasattr(last, "content") else str(last)
    return str(result) if result else ""


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, city="Tokyo")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

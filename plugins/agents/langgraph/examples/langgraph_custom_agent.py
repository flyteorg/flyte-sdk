"""Build your own LangGraph agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a LangGraph
``StateGraph`` using the framework's native SDK, wrap your Flyte tasks as
durable tools with ``tool()``, and pass the compiled graph to
``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as durable LangGraph tool nodes
  that execute as child actions (own container/GPU, retries, caching).
- ``run_agent(agent=...)`` drives the graph loop durably on Flyte.
- The graph definition is fully yours — custom nodes, edges, state, memory, etc.

Run:  flyte run langgraph_custom_agent.py city_agent --city "Tokyo"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.langgraph import run_agent, tool

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


# ── Step 2: Define your own LangGraph agent ──────────────────────────────────
# You control the graph definition — nodes, edges, state, etc.
# Flyte provides the durable runtime via ``tool()`` and ``run_agent(agent=...)``.


def _build_city_graph():
    """Build a LangGraph StateGraph with durable tools."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph.message import MessageGraph

    # Wrap Flyte tasks as durable LangGraph tool nodes
    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")
    population_tool = tool(get_population, name="get_population", description="Get the population of a city.")

    # Use MessageGraph for a simple chat-based graph
    graph = MessageGraph()

    # Add the AI model node
    ChatOpenAI(model="gpt-4o").bind_tools([weather_tool, population_tool])

    # Add tool nodes
    graph.add_node("weather", weather_tool)
    graph.add_node("population", population_tool)

    # Compile the graph
    return graph.compile()


# Build the graph once (in practice, cache this in module scope)
city_graph = _build_city_graph()


# ── Step 3: Run your graph durably inside a Flyte task ───────────────────────
# The graph definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent(city: str) -> str:
    """Run the custom-built graph durably on Flyte."""

    result = await run_agent(
        f"What's the weather in {city}?",
        agent=city_graph,
    )
    return result


# ── Alternative: Pass tools directly to run_agent ────────────────────────────
# If you don't need to separate graph definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    from langgraph.graph import StateGraph

    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")
    population_tool = tool(get_population, name="get_population", description="Get the population of a city.")

    # Build a simple StateGraph
    builder = StateGraph(dict)

    async def ai_node(state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        return {"messages": messages}

    async def tool_node(state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            results = []
            for tc in last.tool_calls:
                tool_fn = None
                for t in [weather_tool, population_tool]:
                    tname = getattr(t, "name", None) or getattr(t, "func", None)
                    if tname and (tname == tc["name"] or (hasattr(tname, "__name__") and tname.__name__ == tc["name"])):
                        tool_fn = t
                        break
                if tool_fn is not None:
                    result = await tool(tool_fn)
                    try:
                        if callable(result):
                            output = result(**tc.get("args", {}))
                            if hasattr(output, "__await__"):
                                output = await output
                        else:
                            output = "No tool available"
                    except Exception as e:
                        output = f"Error: {e}"
                    results.append({"id": tc.get("id", ""), "name": tc["name"], "output": str(output)})
            return {"messages": []}
        return {"messages": []}

    builder.add_node("ai", ai_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("ai")
    builder.add_conditional_edges("ai", lambda s: "tools" if s.get("messages") else "__end__")
    builder.add_edge("tools", "ai")
    agent = builder.compile()

    return await run_agent(
        f"What's the weather in {city}?",
        agent=agent,
    )


# ── Alternative: Build and customize the graph further ───────────────────────
# You can access the compiled graph and add custom nodes or edges.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the graph before running."""
    from langgraph.graph import StateGraph

    builder = StateGraph(dict)

    async def ai_node(state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        return {"messages": messages}

    async def tool_node(state: dict) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            results = []
            for tc in last.tool_calls:
                if tc["name"] == "get_weather":
                    try:
                        output = await tool(get_weather)(city=tc.get("args", {}).get("city", city))
                    except Exception as e:
                        output = f"Error: {e}"
                    results.append({"id": tc.get("id", ""), "name": tc["name"], "output": str(output)})
            return {"messages": []}
        return {"messages": []}

    builder.add_node("ai", ai_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("ai")
    builder.add_conditional_edges("ai", lambda s: "tools" if s.get("messages") else "__end__")
    builder.add_edge("tools", "ai")
    built = builder.compile()

    # You can access and modify the underlying graph here if needed
    # For example, add checkpointing, subgraphs, or human-in-the-loop nodes

    result = await run_agent(
        f"What's the weather in {city}?",
        agent=built,
    )
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, city="Tokyo")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

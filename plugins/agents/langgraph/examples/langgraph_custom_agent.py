"""Build your own LangGraph agent — durable tools on Flyte.

This is the "bring your own graph" path: you build the ``StateGraph`` yourself
with the framework's native SDK, and Flyte provides the durable runtime through
three building blocks from ``flyteplugins.agents.langgraph``:

- ``@tool`` turns a Flyte ``@env.task`` into a first-class LangGraph tool (a
  LangChain ``StructuredTool``) that executes as a durable child action
  (own container/GPU, retries, caching).
- ``ai_node(model, tools)`` is the model-calling node — it binds the tools to
  your chat model and records each model turn durably (replayed on retry).
- ``tool_node(tools)`` is the tool-executing node — it runs the model's tool
  calls as durable Flyte child actions.

You wire them into an ordinary tool-calling loop and compile the graph. Then
``run_agent(agent=compiled_graph)`` drives it inside your task.

Run:  flyte run langgraph_custom_agent.py city_agent --city "Tokyo"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.langgraph import ai_node, run_agent, tool, tool_node

env = flyte.TaskEnvironment(
    "langgraph-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="langgraph-custom-agent")
    .with_local_v2_plugins(["flyteplugins-agents-core", "flyteplugins-agents-langgraph"])
    .with_pip_packages("langchain-openai"),
)


# ── Step 1: Define your durable Flyte tasks and decorate them as tools ────────
# Stacking @tool on @env.task makes each one both a durable, cached Flyte task
# and a first-class LangGraph tool the model can call.


@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@tool
@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


# ── Step 2: Build your own StateGraph ────────────────────────────────────────
# The graph is fully yours — ``ai_node`` and ``tool_node`` are the Flyte-durable
# nodes; you own the state, edges, and control flow. This is the standard
# tool-calling loop: ai → (tools → ai)* → END.


def _build_city_graph():
    """Build a tool-calling StateGraph whose nodes run durably on Flyte."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import tools_condition

    tools = [get_weather, get_population]
    model = ChatOpenAI(model="gpt-4o")

    builder = StateGraph(MessagesState)
    builder.add_node("ai", ai_node(model, tools))
    builder.add_node("tools", tool_node(tools))
    builder.add_edge(START, "ai")
    # ``tools_condition`` routes to the tool node when the model emitted tool
    # calls, otherwise to END.
    builder.add_conditional_edges("ai", tools_condition)
    builder.add_edge("tools", "ai")
    return builder.compile()


# ── Step 3: Run your graph durably inside a Flyte task ────────────────────────
# The graph definition is fully yours; run_agent drives it durably and renders
# the timeline into the task report. The graph is built inside the task, where
# the provider API key is available.


@env.task(report=True, retries=3)
async def city_agent(city: str) -> str:
    """Run the custom-built graph durably on Flyte."""
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        agent=_build_city_graph(),
    )


# ── Alternative: let run_agent build the default tool-calling graph ───────────
# If you don't need a custom topology, pass tools (and a model of your choice)
# and let run_agent assemble the same ai → tools → ai loop for you.


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's default graph builder."""
    from langchain_openai import ChatOpenAI

    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        tools=[get_weather, get_population],
        model=ChatOpenAI(model="gpt-4o"),
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, city="Tokyo")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

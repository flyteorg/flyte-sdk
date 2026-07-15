"""Build your own LangChain agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a LangChain
agent using the framework's native SDK (``langchain.agents.create_agent``),
wrap your Flyte tasks as durable tools with ``tool()``, and pass the built
agent to ``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as durable LangChain tools
  (real ``StructuredTool``s) that execute as child actions (own container/GPU,
  retries, caching).
- ``create_agent(model, tools, system_prompt=...)`` returns a compiled LangGraph
  graph — your agent, fully yours (custom prompt, model, etc.).
- ``run_agent(agent=...)`` drives that graph durably on Flyte.

Run:  flyte run langchain_custom_agent.py city_agent_task --city "Paris"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.langchain import run_agent, tool

env = flyte.TaskEnvironment(
    "langchain-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="langchain-custom-agent")
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
                package_name="flyteplugins-agents-langchain",
                pre=True,
            ),
        )
    ),
)


# ── Step 1: Define your durable Flyte tasks, wrapped as tools ────────────────
# Stack @tool on top of @env.task: each is both a durable, cached Flyte task and
# a LangChain tool the agent can call. Wrapping at module scope lets the same
# tool objects be reused across the agent build and any direct run_agent call.


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


# ── Step 2: Define your own LangChain agent ──────────────────────────────────
# You control the agent definition — prompt, model, tools. In langchain 1.x this
# is ``create_agent``, which returns a compiled graph. Flyte provides the durable
# runtime via ``tool()`` and ``run_agent(agent=...)``.


def _build_city_agent():
    """Build a LangChain agent (compiled graph) with durable tools."""
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    return create_agent(
        ChatOpenAI(model="gpt-4o"),
        tools=[get_weather, get_population],
        system_prompt=(
            "You are a helpful city-facts assistant. Use the available tools to answer "
            "questions about cities. Be concise and accurate."
        ),
    )


# Build the agent once at module scope.
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the graph durably. The
# agent already owns its tools, so do NOT pass `tools=` alongside `agent=`.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    return await run_agent(
        f"What's the weather in {city}?",
        agent=city_agent,
    )


# ── Alternative: let run_agent build the agent from tools ────────────────────
# If you don't need to separate agent definition from execution, hand the tools
# (and optional instructions/model) straight to run_agent's builder path.


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    return await run_agent(
        f"What's the weather in {city} and how many people live there?",
        tools=[get_weather, get_population],
        instructions="You are a helpful city-facts assistant.",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="Paris")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

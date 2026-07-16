"""Build your own CrewAI agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a CrewAI
``Agent`` using the framework's native SDK, attach your Flyte tasks as durable
tools via ``Agent(tools=[...])``, and pass the agent to ``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as native CrewAI ``BaseTool``
  instances that execute as durable child actions (own container/GPU, retries,
  caching). Because they are real ``BaseTool``s, they attach directly to
  ``Agent(tools=[...])`` — exactly like any hand-written CrewAI tool.
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte. When you pass a
  pre-built ``agent`` it already carries its own tools, so do NOT also pass
  ``tools=`` (that raises ``ValueError``).
- The agent definition is fully yours — custom roles, goals, backstory, etc.

Run:  flyte run crewai_custom_agent.py city_agent_task --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.crewai import run_agent, tool

env = flyte.TaskEnvironment(
    "crewai-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="crewai-custom-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-crewai"]
    ),
)


# ── Step 1: Define your durable Flyte tasks as tools ─────────────────────────
# Stack @tool on top of @env.task: each is both a normal, durable, cached Flyte
# task AND a native CrewAI BaseTool you can attach to an Agent. Every tool call
# the agent makes runs as a durable Flyte child action.


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


# ── Step 2: Define your own CrewAI agent ─────────────────────────────────────
# You control the agent definition — role, goal, backstory, tools, etc. The
# durable tools attach natively via ``tools=[...]`` just like any CrewAI tool.


def _build_city_agent():
    """Build a CrewAI Agent with durable Flyte tools attached natively."""
    from crewai import Agent

    return Agent(
        role="City Facts Expert",
        goal="Provide accurate and concise information about cities worldwide.",
        backstory=(
            "You are a knowledgeable city-facts assistant with expertise in urban "
            "demographics, climate, and geography. You answer questions helpfully."
        ),
        tools=[get_weather, get_population],
        llm="gpt-4o",
    )


# Build the agent once (in practice, cache this in module scope)
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably. The
# pre-built agent already carries its tools, so we pass ``agent=`` only — never
# ``tools=`` alongside it.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        agent=city_agent,
    )


# ── Alternative: let run_agent build the agent from tools ─────────────────────
# If you don't need to hand-craft the Agent, pass the durable tools to run_agent
# and it builds and drives an agent for you (no separate ``agent=``).


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="gpt-4o",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

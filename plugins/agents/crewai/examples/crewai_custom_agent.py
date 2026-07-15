"""Build your own CrewAI agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a CrewAI
``Agent`` using the framework's native SDK, wrap your Flyte tasks as durable
tools with ``tool()``, and pass the agent to ``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as durable CrewAI tools
  that execute as child actions (own container/GPU, retries, caching).
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.
- The agent definition is fully yours — custom roles, goals, backstory, etc.

Run:  flyte run crewai_custom_agent.py city_agent --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.crewai import run_agent, tool

env = flyte.TaskEnvironment(
    "crewai-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="crewai-custom-agent")
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
                package_name="flyteplugins-agents-crewai",
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


# ── Step 2: Define your own CrewAI agent ─────────────────────────────────────
# You control the agent definition — role, goal, backstory, tools, etc.
# Flyte provides the durable runtime via ``tool()`` and ``run_agent(agent=...)``.


def _build_city_agent():
    """Build a CrewAI Agent with durable tools."""
    from crewai import Agent

    # Create the agent with your custom configuration
    return Agent(
        role="City Facts Expert",
        goal="Provide accurate and concise information about cities worldwide.",
        backstory=(
            "You are a knowledgeable city-facts assistant with expertise in urban "
            "demographics, climate, and geography. You answer questions helpfully."
        ),
        llm="gpt-4o",
    )


# Build the agent once (in practice, cache this in module scope)
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    result = await run_agent(
        f"What's the weather in {city}?",
        agent=city_agent,
    )
    return result


# ── Alternative: Pass tools directly to run_agent ────────────────────────────
# If you don't need to separate agent definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    from crewai import Agent

    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")
    population_tool = tool(get_population, name="get_population", description="Get the population of a city.")

    # Build the agent with tools
    agent = Agent(
        role="Assistant",
        goal="Answer the user's question accurately and concisely.",
        backstory="You are a helpful assistant.",
        llm="gpt-4o",
    )

    return await run_agent(
        f"What's the weather in {city}?",
        agent=agent,
        tools=[weather_tool, population_tool],
    )


# ── Alternative: Pre-build with custom role and goal ─────────────────────────
# You can customize the agent with specific roles and goals for different tasks.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the agent with specific role and goal before running."""
    from crewai import Agent

    weather_tool = tool(get_weather, name="get_weather", description="Get the current weather for a city.")

    # Build the agent with custom configuration
    agent = Agent(
        role="Weather Specialist",
        goal="Provide detailed weather information including temperature, conditions, and forecasts.",
        backstory=(
            "You are a weather expert with deep knowledge of climate patterns, "
            "meteorology, and environmental science. Always include temperature and conditions."
        ),
        llm="gpt-4o",
    )

    # You can access and modify the underlying agent here if needed
    # For example, add additional tools or modify the model

    result = await run_agent(
        f"What's the weather in {city}?",
        agent=agent,
        tools=[weather_tool],
    )
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

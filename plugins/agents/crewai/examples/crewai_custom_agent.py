"""Build your own CrewAI agent — durable tools on Flyte.

This example shows the "bring your own agent SDK" path: you write your own
CrewAI ``Agent`` with custom roles, goals, and tool bindings, and Flyte
provides the durable runtime underneath.

- ``CrewAIAgent.durable_tools()`` wraps your Flyte tasks so each tool call
  executes as a durable child action (own container/GPU, retries, caching).
- ``CrewAIAgent.build()`` returns a ``crewai.Agent`` you can customize further
  (configurable roles, goals, backstory, etc.).
- The agent runs durably inside a Flyte task — every tool call is a
  first-class Flyte node.

Run:  flyte run crewai_custom_agent.py city_agent --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.crewai import CrewAIAgent

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


# ── Step 2: Build your own agent with CrewAIAgent ────────────────────────────
# You control the agent definition — role, goal, backstory, tools, etc.
# CrewAIAgent gives you durable tools and a clean builder.


agent = CrewAIAgent(
    model="gpt-4o",
    name="city-facts-agent",
    role="City Facts Expert",
    goal="Provide accurate and concise information about cities worldwide.",
    backstory=(
        "You are a knowledgeable city-facts assistant with expertise in urban "
        "demographics, climate, and geography. You answer questions helpfully."
    ),
)

# Wrap your Flyte tasks as durable CrewAI tools
tools = agent.durable_tools(get_weather, get_population)

# Build the agent — this returns a crewai.Agent you can customize further
built_agent = agent.build(tools=tools)


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; the task is the durable runtime.


@env.task(report=True, retries=3)
async def city_agent(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    result = await built_agent.run(input=f"What's the weather in {city}?")
    return result


# ── Alternative: Use CrewAIAgent.run() for a quick build-and-run ─────────────
# If you don't need to separate agent definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call."""
    quick_agent = CrewAIAgent("gpt-4o", name="quick-agent")
    durable_tools = quick_agent.durable_tools(get_weather, get_population)
    return await quick_agent.run(
        f"What's the weather in {city}?",
        tools=durable_tools,
    )


# ── Alternative: Pre-build with custom role and goal ─────────────────────────
# You can customize the agent with specific roles and goals for different tasks.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the agent with specific role and goal before running."""
    agent = CrewAIAgent(
        model="gpt-4o",
        name="weather-expert",
        role="Weather Specialist",
        goal="Provide detailed weather information including temperature, conditions, and forecasts.",
        backstory=(
            "You are a weather expert with deep knowledge of climate patterns, "
            "meteorology, and environmental science. Always include temperature and conditions."
        ),
    )
    built = agent.build(tools=agent.durable_tools(get_weather))

    # You can access and modify the underlying agent here if needed
    # For example, add additional tools or modify the model

    result = await built.run(input=f"What's the weather in {city}?")
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

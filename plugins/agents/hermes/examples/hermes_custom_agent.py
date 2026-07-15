"""Build your own Hermes agent — durable tools on Flyte.

This example shows the "bring your own agent SDK" path: you write your own
Hermes ``Agent`` with custom prompts, models, and tool bindings, and Flyte
provides the durable runtime underneath.

- ``FlyteAgent.durable_tools()`` wraps your Flyte tasks so each tool call
  executes as a durable child action (own container/GPU, retries, caching).
- ``FlyteAgent.build()`` returns a ``hermes.Agent`` you can customize further
  (hand-offs, retries, custom models, etc.).
- The agent runs durably inside a Flyte task — every tool call is a
  first-class Flyte node.

Run:  flyte run hermes_custom_agent.py city_agent --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.hermes import FlyteAgent

env = flyte.TaskEnvironment(
    "hermes-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="hermes-custom-agent")
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
                package_name="flyteplugins-agents-hermes",
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
# You control the agent definition — prompt, model, tools, etc.
# FlyteAgent gives you durable tools and a clean builder.


agent = FlyteAgent(
    model="gpt-4o",
    name="city-facts-agent",
    system_prompt=(
        "You are a helpful city-facts assistant. Use the available tools to answer "
        "questions about cities. Be concise and accurate."
    ),
)

# Wrap your Flyte tasks as durable Hermes tools
tools = agent.durable_tools(get_weather, get_population)

# Build the agent — this returns a hermes.Agent you can customize further
built_agent = agent.build(tools=tools)


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; the task is the durable runtime.


@env.task(report=True, retries=3)
async def city_agent(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    result = await built_agent.run(f"What's the weather in {city}?")
    return result


# ── Alternative: Use FlyteAgent.run() for a quick build-and-run ──────────────
# If you don't need to separate agent definition from execution:


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call."""
    quick_agent = FlyteAgent("gpt-4o", name="quick-agent")
    durable_tools = quick_agent.durable_tools(get_weather, get_population)
    return await quick_agent.run(
        f"What's the weather in {city}?",
        tools=durable_tools,
    )


# ── Alternative: Pre-build and customize the agent further ───────────────────
# You can also get the raw hermes.Agent and modify it before running.


@env.task(report=True, retries=3)
async def custom_city_agent(city: str) -> str:
    """Customize the agent before running."""
    agent = FlyteAgent(
        model="gpt-4o",
        name="custom-agent",
        system_prompt="You are a weather expert. Always include temperature and conditions.",
    )
    built = agent.build(tools=agent.durable_tools(get_weather))

    # You can access and modify the underlying agent here if needed
    # For example, add additional tools or modify the model

    result = await built.run(f"What's the weather in {city}?")
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

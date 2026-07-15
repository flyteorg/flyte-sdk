"""Build your own Hermes agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a Hermes
``Agent`` using the framework's native SDK, wrap your Flyte tasks as durable
tools with ``@tool``, attach them to the agent (the framework convention), and
pass the agent to ``run_agent(agent=...)``.

- ``@tool`` stacked on ``@env.task`` makes each function both a durable, cached
  Flyte task and a Hermes tool that executes as a child action (own
  container/GPU, retries, caching).
- ``_build_city_agent()`` attaches the tools natively via ``Agent(tools=[...])``.
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.

Run:  flyte run hermes_custom_agent.py city_agent_task --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.hermes import run_agent, tool

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


# ── Step 1: Define your durable Flyte tasks and decorate them as tools ────────
# Stacking @tool on @env.task makes each one both a durable, cached Flyte task
# and a Hermes tool the agent can call.


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


# ── Step 2: Define your own Hermes agent with the tools attached ──────────────
# You control the agent definition — prompt, model, tools, etc. Flyte provides
# the durable runtime via ``@tool`` and ``run_agent(agent=...)``.


def _build_city_agent():
    """Build a Hermes Agent with durable tools attached natively."""
    from hermes import Agent

    return Agent(
        model="gpt-4o",
        name="city-facts-agent",
        system_prompt=(
            "You are a helpful city-facts assistant. Use the available tools to answer "
            "questions about cities. Be concise and accurate."
        ),
        tools=[get_weather, get_population],
    )


# Build the agent once (module scope, reused across runs).
city_agent = _build_city_agent()


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    return await run_agent(
        f"What's the weather and population of {city}?",
        agent=city_agent,
    )


# ── Alternative: let run_agent build the agent from tools ─────────────────────
# If you don't need to separate agent definition from execution, hand the tools
# to run_agent and it builds the agent for you.


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    return await run_agent(
        f"What's the weather and population of {city}?",
        tools=[get_weather, get_population],
        instructions="You are a helpful city-facts assistant.",
        model="gpt-4o",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent_task, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

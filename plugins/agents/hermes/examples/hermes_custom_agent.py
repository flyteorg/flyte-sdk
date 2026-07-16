"""Build your own Hermes agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you configure Hermes'
``AIAgent`` using the framework's native SDK, wrap your Flyte tasks as durable
tools with ``@tool`` (which registers them in Hermes' tool registry under the
``FLYTE_TOOLSET`` toolset — the framework convention), and pass the agent to
``run_agent(agent=...)``.

- ``@tool`` stacked on ``@env.task`` makes each function both a durable, cached
  Flyte task and a Hermes tool that executes as a child action (own
  container/GPU, retries, caching).
- ``_build_city_agent()`` enables the tools natively via
  ``AIAgent(enabled_toolsets=[FLYTE_TOOLSET])``. The agent is built inside the
  task because Hermes resolves credentials at construction time.
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.

Run:  flyte run hermes_custom_agent.py city_agent_task --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os

import flyte

from flyteplugins.agents.hermes import FLYTE_TOOLSET, run_agent, tool

env = flyte.TaskEnvironment(
    "hermes-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="hermes-custom-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-hermes"]
    ),
)


# ── Step 1: Define your durable Flyte tasks and decorate them as tools ────────
# Stacking @tool on @env.task makes each one both a durable, cached Flyte task
# and a Hermes tool the agent can call (registered under FLYTE_TOOLSET).


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


# ── Step 2: Define your own Hermes agent with the tools enabled ───────────────
# You control the agent definition — prompt, model, toolsets, etc. Flyte
# provides the durable runtime via ``@tool`` and ``run_agent(agent=...)``.


def _build_city_agent():
    """Build a Hermes AIAgent with the Flyte toolset enabled natively."""
    from run_agent import AIAgent  # hermes-agent's top-level module

    return AIAgent(
        model="gpt-4o",
        ephemeral_system_prompt=(
            "You are a helpful city-facts assistant. Use the available tools to answer "
            "questions about cities. Be concise and accurate."
        ),
        enabled_toolsets=[FLYTE_TOOLSET],
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1",
        quiet_mode=True,
    )


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably.


@env.task(report=True, retries=3)
async def city_agent_task(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    # Built in-task: Hermes reads credentials when the agent is constructed.
    city_agent = _build_city_agent()
    return await run_agent.aio(
        f"What's the weather and population of {city}?",
        agent=city_agent,
    )


# ── Alternative: let run_agent build the agent from tools ─────────────────────
# If you don't need to separate agent definition from execution, hand the tools
# to run_agent and it builds the agent for you (model= is required).


@env.task(report=True, retries=3)
async def quick_city_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder."""
    return await run_agent.aio(
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

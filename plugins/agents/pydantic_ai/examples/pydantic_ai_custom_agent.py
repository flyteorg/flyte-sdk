"""Build your own Pydantic AI agent — durable tools on Flyte.

This example shows the "bring your own agent" path: you define a Pydantic AI
``Agent`` using the framework's native SDK, wrap your Flyte tasks as durable
tools with ``tool()``, and pass the agent to ``run_agent(agent=...)``.

- ``tool()`` wraps Flyte ``@env.task`` functions as durable Pydantic AI tools
  that execute as child actions (own container/GPU, retries, caching).
- ``run_agent(agent=...)`` drives the agent loop durably on Flyte.
- The agent definition is fully yours — custom prompts, models, hand-offs, etc.

Run:  flyte run pydantic_ai_custom_agent.py weather_agent --city "San Francisco"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import functools

import flyte

from flyteplugins.agents.pydantic_ai import run_agent, tool

env = flyte.TaskEnvironment(
    "pydantic-ai-custom-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="pydantic-ai-custom-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-pydantic-ai"]
    ),
)


# ── Step 1: Define your durable Flyte tasks as tools ─────────────────────────
# Stack ``@tool`` on top of ``@env.task``: each is both a normal, durable, cached
# Flyte task AND a Pydantic AI tool you can hand to an ``Agent`` natively.


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


# ── Step 2: Define your own Pydantic AI agent ────────────────────────────────
# You control the agent definition — prompt, model, tools, etc. The tools are
# attached natively via ``Agent(tools=[...])``; Flyte makes each tool call durable.


@functools.lru_cache
def _build_city_agent():
    """Build a Pydantic AI Agent with durable Flyte-task tools attached natively.

    Built lazily (and cached) rather than at module scope: constructing an
    ``Agent("openai:gpt-4o", ...)`` eagerly validates the OpenAI provider (it needs
    an API key), so building at import time would force ``OPENAI_API_KEY`` into the
    LOCAL env just to ``flyte run`` this file. Call it inside the task body, where
    the ``openai_api_key`` secret is present.
    """
    from pydantic_ai import Agent

    return Agent(
        "openai:gpt-4o",
        name="city-facts-agent",
        system_prompt=(
            "You are a helpful city-facts assistant. Use the available tools to answer "
            "questions about cities. Be concise and accurate."
        ),
        tools=[get_weather, get_population],
    )


# ── Step 3: Run your agent durably inside a Flyte task ───────────────────────
# The agent definition is fully yours; run_agent drives the loop durably. The
# tools already live on the agent, so pass ``agent=`` with NO separate ``tools=``.


@env.task(report=True, retries=3)
async def weather_agent(city: str) -> str:
    """Run the custom-built agent durably on Flyte."""
    agent = _build_city_agent()
    return await run_agent.aio(
        f"What's the weather in {city}?",
        agent=agent,
    )


# ── Alternative: let run_agent build the agent from tools ─────────────────────
# If you don't want to define the ``Agent`` yourself, pass ``tools=`` + ``model=``
# (and NO ``agent=``) and run_agent builds the agent for you.


@env.task(report=True, retries=3)
async def quick_weather_agent(city: str) -> str:
    """Build and run in one call using run_agent's builder path."""
    return await run_agent.aio(
        f"What's the weather in {city}?",
        tools=[get_weather, get_population],
        model="openai:gpt-4o",
        instructions="You are a helpful city-facts assistant. Be concise and accurate.",
        name="quick-agent",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(weather_agent, city="San Francisco")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

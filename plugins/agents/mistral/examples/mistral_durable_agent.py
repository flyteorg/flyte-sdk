"""Run a Mistral Agents (Conversations API) agent durably on Flyte.

Mistral runs the agent's reasoning server-side; Flyte is the runtime underneath:

- Each tool is a Flyte task — it runs as a durable child action (own
  container/resources, retries, caching) when the agent calls it.
- Each conversation turn is recorded via ``flyte.trace`` (it's an in-process HTTP
  call), so a crash/retry replays completed turns + cache-hits tools — Mistral
  gets both per-turn and per-tool durability.
- The agent timeline renders into the task report (``report=True``).
- ``run_agent`` is syncified: ``await run_agent.aio(...)`` from async tasks, plain
  ``run_agent(...)`` from sync tasks (see ``city_agent_sync``).

Run:  flyte run mistral_durable_agent.py city_agent --question "What's the weather and population of Paris?"
      (or drive the sync variant:  flyte run mistral_durable_agent.py city_agent_sync --question "...")
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.mistral import run_agent, tool

env = flyte.TaskEnvironment(
    "mistral-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="mistral_api_key", as_env_var="MISTRAL_API_KEY")],
    image=flyte.Image.from_debian_base(name="mistral-durable-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-mistral"]
    ),
)


# Stack @tool on top of @env.task: each is both a Mistral tool and a
# normal, durable, cached Flyte task.
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


@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    """The durable parent: the Mistral agent loop runs here, on Flyte."""
    return await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="mistral-large-latest",
    )


# ``run_agent`` is syncified: async tasks await ``run_agent.aio(...)`` (above),
# while sync tasks simply call ``run_agent(...)``.
@env.task(report=True, retries=3)
def city_agent_sync(question: str) -> str:
    """The same agent, driven from a sync task via the sync call form."""
    return run_agent(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="mistral-large-latest",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

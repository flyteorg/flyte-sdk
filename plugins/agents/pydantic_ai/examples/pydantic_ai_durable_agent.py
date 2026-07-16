"""Run a Pydantic AI agent durably on Flyte (single agent).

The "bring your own agent SDK" path: you keep writing Pydantic AI code,
and Flyte is the runtime underneath.

- Each tool is a Flyte task — it runs as a durable child action (its own
  container/resources, with retries and caching) when the agent calls it.
- The agent timeline (tool calls, AI messages) is rendered into the task report
  because the task is created with ``report=True``.
- ``run_agent`` is syncified: ``await run_agent.aio(...)`` from async tasks, plain
  ``run_agent(...)`` from sync tasks (see ``city_agent_sync``).

Run:  flyte run pydantic_ai_durable_agent.py city_agent --question "What's the weather of Paris?"
      (or drive the sync variant:  flyte run pydantic_ai_durable_agent.py city_agent_sync --question "...")
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.pydantic_ai import run_agent, tool

env = flyte.TaskEnvironment(
    "pydantic-ai-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="pydantic-ai-durable-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-pydantic-ai"]
    ),
)


# Stack @tool on top of @env.task: each is both a tool the agent can
# call and a normal, durable, cached Flyte task.
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
    """The durable parent: the Pydantic AI agent loop runs here, on Flyte."""
    return await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="openai:gpt-4o",
        name="city-agent",
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
        model="openai:gpt-4o",
        name="city-agent",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

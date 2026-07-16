"""Run an OpenAI Agents SDK agent durably on Flyte (single agent).

The "bring your own agent SDK" path: you keep writing OpenAI Agents SDK code,
and Flyte is the runtime underneath.

- Each tool is a Flyte task — it runs as a durable child action (its own
  container/resources, with retries and caching) when the agent calls it.
- Each model turn is recorded via ``flyte.trace``, so if ``city_agent`` crashes
  and Flyte retries it (``retries=3``), completed turns and tool calls are
  replayed instead of re-calling (and re-billing) the model — self-healing.
- Intermittent model failures (429s, 5xx, timeouts) are retried in place by the
  OpenAI client (``max_retries``), below the durable wrapper, so a turn is
  recorded only once it succeeds.
- The agent timeline (turns, tool calls, token usage) is rendered into the task
  report because the task is created with ``report=True``.
- ``run_agent`` is syncified: ``await run_agent.aio(...)`` from async tasks, plain
  ``run_agent(...)`` from sync tasks (see ``city_agent_sync``).

Run:  flyte run openai_durable_agent.py city_agent --question "What's the weather and population of Paris?"
      (or drive the sync variant:  flyte run openai_durable_agent.py city_agent_sync --question "...")
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte
from agents import OpenAIProvider, RunConfig
from openai import AsyncOpenAI

from flyteplugins.agents.openai import run_agent, tool

env = flyte.TaskEnvironment(
    "openai-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="openai-durable-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-openai"]
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
    """The durable parent: the OpenAI agent loop runs here, on Flyte."""
    client = AsyncOpenAI(max_retries=5, timeout=30)
    return await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="gpt-4.1",
        run_config=RunConfig(model_provider=OpenAIProvider(openai_client=client)),
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
        model="gpt-4.1",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

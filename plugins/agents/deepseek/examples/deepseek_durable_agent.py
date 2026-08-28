"""Run a DeepSeek Harness agent durably on Flyte (single agent).

The "bring your own agent SDK" path: you keep writing DeepSeek Harness code, and
Flyte is the runtime underneath.

- Each tool is a Flyte task — it runs as a durable child action (its own
  container/resources, with retries and caching) when the agent calls it.
- The agent timeline (assistant turns, tool calls and their results) is rendered
  into the task report because the task is created with ``report=True``.
- ``run_agent`` is async: ``await run_agent(...)`` from async tasks, while sync
  tasks call ``run_agent_sync(...)`` (see ``city_agent_sync``).

Because the harness has no tool-registration channel of its own, each tool is
published into the harness workspace as an executable command and described to
the model in its prompt; running it calls straight back into this task. See the
adapter README for the mechanism.

Run:  flyte run deepseek_durable_agent.py city_agent --question "What's the weather of Paris?"
      (or drive the sync variant:  flyte run deepseek_durable_agent.py city_agent_sync --question "...")
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import flyte

from flyteplugins.agents.deepseek import run_agent, run_agent_sync, tool

env = flyte.TaskEnvironment(
    "deepseek-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="deepseek_api_key", as_env_var="DEEPSEEK_API_KEY")],
    image=flyte.Image.from_debian_base(name="deepseek-durable-agent").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-deepseek"]
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
    """The durable parent: the DeepSeek Harness agent loop runs here, on Flyte."""
    return await run_agent(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the provided tools to answer.",
        model="deepseek-v4-flash",
    )


# ``run_agent`` is async: async tasks await ``run_agent(...)`` (above),
# while sync tasks call ``run_agent_sync(...)``.
@env.task(report=True, retries=3)
def city_agent_sync(question: str) -> str:
    """The same agent, driven from a sync task via the sync call form."""
    return run_agent_sync(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the provided tools to answer.",
        model="deepseek-v4-flash",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

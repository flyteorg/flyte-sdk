"""Run a Claude Agent SDK agent durably on Flyte.

The Claude Agent SDK owns the loop (it drives the model via the Claude Code
runtime); Flyte is the runtime underneath:

- Each tool is a Flyte task — it runs as a durable child action (own
  container/resources, retries, caching) when Claude calls it.
- The agent timeline (assistant turns, tool calls, cost) renders into the task
  report because the task is created with ``report=True``.

Per-turn model replay isn't available for Claude (the model loop runs in the
Claude Code runtime), but tool calls are durable regardless.

The ``claude-agent-sdk`` wheel bundles the native ``claude`` CLI, so the image needs
no separate Node.js / ``@anthropic-ai/claude-code`` install — just an Anthropic API key.

Run:  python claude_durable_agent.py
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.claude import function_tool, run_agent

# The Claude Agent SDK bundles the native `claude` CLI in its wheel, so the image
# only needs the adapter — installed here from locally-built wheels under `../dist`.
env = flyte.TaskEnvironment(
    "claude-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="claude-durable-agent").clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-core",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-claude",
                pre=True,
            ),
        )
    ),
)


# Stack @function_tool on top of @env.task: each is both a Claude tool and a
# normal, durable, cached Flyte task.
@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22°C."


@function_tool
@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


@env.task(report=True, retries=3)
async def city_agent(question: str) -> str:
    """The durable parent: the Claude agent loop runs here, on Flyte."""
    return await run_agent(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="claude-sonnet-4-5",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

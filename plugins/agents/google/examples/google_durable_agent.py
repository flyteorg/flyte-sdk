"""Run a Google ADK (Agent Development Kit) agent durably on Flyte.

ADK's ``Runner`` owns the agent loop (drives the model + tools and yields events);
Flyte is the runtime underneath:

- Each tool is a Flyte task — a durable child action (own container/resources,
  retries, caching) when the agent calls it.
- Each model turn is recorded via ``flyte.trace`` (the ``FlyteLlm`` seam below the
  loop), so a crash/retry replays completed turns and cache-hits tools.
- The agent timeline (turns, tool calls) renders into the task report (``report=True``).

Set a Gemini API key (``GOOGLE_API_KEY``) as a Flyte secret.

Run:  python google_durable_agent.py
"""

from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.google import function_tool, run_agent

env = flyte.TaskEnvironment(
    "google-durable-agent",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="google-durable-agent").clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-core",
                pre=True,
            ),
        )
        .clone(
            addl_layer=PythonWheels(
                wheel_dir=Path(__file__).parent.parent / "dist",
                package_name="flyteplugins-agents-google",
                pre=True,
            ),
        )
    ),
)


# Stack @function_tool on top of @env.task: each is both an ADK tool and a normal,
# durable, cached Flyte task.
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
    """The durable parent: the Google ADK agent loop runs here, on Flyte."""
    return await run_agent(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="gemini-2.0-flash",
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

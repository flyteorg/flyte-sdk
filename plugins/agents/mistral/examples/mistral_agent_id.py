"""Drive a pre-created server-side Mistral agent (by id) durably on Flyte.

Mistral lets you create a persistent, server-side agent — its model,
instructions and guardrails live in Mistral and are managed in one place. This
example creates such an agent once, then runs it on Flyte by ``agent_id`` instead
of an inline ``model``, while still attaching Flyte-task tools, so the
server-side agent's tool calls are durable child actions.

    create_agent             ->  ag_xxx  (server-side, reusable)
    run_with_agent(ag_xxx)   ->  agent runs on Flyte; tools are durable actions

In practice you create the agent once (here, or in the Mistral console) and reuse
its id across many runs — pass an existing ``agent_id`` to skip creation.

Run:  flyte run mistral_agent_id.py city_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os
from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.mistral import function_tool, run_agent

env = flyte.TaskEnvironment(
    "mistral-agent-id",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="mistral_api_key", as_env_var="MISTRAL_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="mistral-agent-id")
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
                package_name="flyteplugins-agents-mistral",
                pre=True,
            ),
        )
    ),
)


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


@env.task(retries=3)
async def create_agent(name: str, instructions: str, model: str = "mistral-large-latest") -> str:
    """Create a reusable server-side Mistral agent; return its id.

    The API key is read from the environment (a Flyte secret), never an argument.
    """
    from mistralai.client import Mistral

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    agent = await client.beta.agents.create_async(model=model, name=name, instructions=instructions)
    return agent.id


@env.task(report=True, retries=3)
async def run_with_agent(question: str, agent_id: str) -> str:
    """Run a server-side agent (by id) on Flyte, with Flyte-backed tools.

    ``agent_id`` is passed instead of ``model`` — ``run_agent`` drives the existing
    server-side agent and registers our Flyte tasks as its tools.
    """
    return await run_agent(question, tools=[get_weather, get_population], agent_id=agent_id)


@env.task(retries=3)
async def city_agent(question: str) -> str:
    """Create the server-side agent, then run it — durably, on Flyte."""
    agent_id = await create_agent(
        name="city-facts",
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
    )
    return await run_with_agent(question, agent_id)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(city_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

"""Crash-and-resume: durable CrewAI agent recovery on Flyte.

Shows that a crash mid-run does not redo completed work. On the first attempt
the agent does real work (model turns + tool calls), then the worker is killed
(simulated). Flyte retries the task; on the second attempt the completed model
turns replay from their ``flyte.trace`` records and the tool calls are cache
hits, so the agent finishes without re-calling (or re-billing) the model.

How the replay shows up here: ``run_agent``'s builder drives the agent with a
durable ``crewai.LLM`` — every completed model turn is recorded through the
shared durable step before it returns. On the retry those turns are served from
their trace records and CrewAI's LLM is never re-invoked (no injectable model
hook is needed — durability lives inside the LLM the builder installs). The
visible proof is the tools: on attempt 0 each prints an "EXECUTED ... cache
MISS" line; on the retry those lines are absent because every call is a cache
hit, and the run completes near-instantly.

Run this on a Flyte / Union backend, where attempt numbers and durable trace
records are provided per attempt — that is where the replay is visible. In
``local`` mode Flyte still retries, but does not persist trace records across
attempts and does not set a per-attempt number, so this example simply runs once
locally (the crash is skipped) and the replay is not exercised.

Run:  flyte run crewai_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os

import flyte

from flyteplugins.agents.crewai import run_agent, tool

env = flyte.TaskEnvironment(
    "crewai-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="crewai-crash-resume").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-crewai"]
    ),
)


@tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    print(f"  🛠  get_weather EXECUTED for {city} (cache MISS)", flush=True)
    return f"sunny, 22°C in {city}"


@tool
@env.task(cache="auto", retries=3)
async def get_population(city: str) -> int:
    """Get the population of a city."""
    print(f"  🛠  get_population EXECUTED for {city} (cache MISS)", flush=True)
    return {"San Francisco": 808988, "Paris": 2102650, "Tokyo": 13929286}.get(city, 1_000_000)


@env.task(report=True, retries=2)
async def resilient_agent(question: str) -> str:
    attempt = flyte.ctx().attempt_number if flyte.ctx() else 0
    print(f"▶ resilient_agent attempt {attempt}", flush=True)

    # durable=True (the default) makes the builder install a durable LLM: each
    # model turn is recorded via flyte.trace, so on the retry completed turns
    # replay from their records and CrewAI never re-calls the model.
    answer = await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="gpt-4o",
    )

    # Simulate a worker crash after the agent did real work, but only on the
    # first attempt on a backend.
    on_backend = os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None
    if on_backend and attempt == 0:
        raise RuntimeError("💥 simulated worker crash (first attempt only)")

    print("✅ completed on retry — model turns replayed, tools cache-hit", flush=True)
    return answer


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(resilient_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

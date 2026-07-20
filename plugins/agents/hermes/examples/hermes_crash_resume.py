"""Crash-and-resume: durable agent recovery on Flyte (tool-level for Hermes).

Shows that a crash mid-run does not redo completed tool work. On the first
attempt the agent does real work (model turns + tool calls), then the worker is
killed (simulated). Flyte retries the task; on the second attempt the completed
tool calls are cache HITs (their tasks do not re-execute — the "EXECUTED" lines
below are absent), so the run self-heals without redoing tool work.

Honesty note — what is and isn't replayed: unlike the openai/langchain
adapters, ``hermes-agent`` exposes no per-model-turn hook (the model client is
buried inside ``AIAgent``), so completed model turns are NOT replayed from
``flyte.trace`` records — the retry re-drives the model (and re-bills those
turns). Durability for Hermes is at tool granularity: each ``@tool`` task is a
durable Flyte child action with retries and caching, and the enclosing task's
``retries=`` provides the self-healing.

Run this on a Flyte / Union backend, where attempt numbers and the task cache
are provided per attempt — that is where the recovery is visible. In ``local``
mode Flyte does not set a per-attempt number, so this example simply runs once
locally (the crash is skipped).

Run:  flyte run hermes_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os

import flyte

from flyteplugins.agents.hermes import run_agent, tool

env = flyte.TaskEnvironment(
    "hermes-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="hermes-crash-resume").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-hermes"]
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

    answer = await run_agent(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model="gpt-4.1",
    )

    # Simulate a worker crash after the agent did real work, but only on the
    # first attempt on a backend.
    on_backend = os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None
    if on_backend and attempt == 0:
        raise RuntimeError("💥 simulated worker crash (first attempt only)")

    print("✅ completed on retry — tools cache-hit; model turns re-driven (no per-turn hook)", flush=True)
    return answer


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(resilient_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

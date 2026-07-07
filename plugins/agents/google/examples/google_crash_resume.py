"""Crash-and-resume: durable Google ADK agent recovery on Flyte.

Shows that a crash mid-run does not redo completed work. On the first attempt the
agent does real work (model turns + tool calls), then the worker is killed (simulated).
Flyte retries the task; on the second attempt:

- completed model turns replay — with ``durable=True`` (the default) the agent's
  model is wrapped (``FlyteLlm``) so each turn through ``BaseLlm.generate_content_async``
  — the seam below the loop — is recorded via ``flyte.trace`` and replayed instead of
  re-calling (and re-billing) the model;
- completed tool calls are cache hits — each tool is a durable Flyte child action
  with ``cache="auto"``, so it isn't re-executed.

Backend only: per-attempt numbers and durable trace records are provided per attempt.
In ``local`` mode the crash is skipped and the example just runs once.

Run:  flyte run google_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os
from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.google import tool, run_agent

env = flyte.TaskEnvironment(
    "google-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="google_api_key", as_env_var="GOOGLE_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="google-crash-resume")
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
                package_name="flyteplugins-agents-google",
                pre=True,
            ),
        )
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
        model="gemini-2.5-flash-lite",
        durable=True,  # default — record each model turn for replay on retry
    )

    # Simulate a worker crash AFTER the agent did real work, but only on the first
    # attempt on a backend. ``FLYTE_ATTEMPT_NUMBER`` is only set per attempt on a
    # backend, so local runs skip the crash and just complete.
    on_backend = os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None
    if on_backend and attempt == 0:
        raise RuntimeError("💥 simulated worker crash (first attempt only)")

    print("✅ completed on retry — turns replayed, tools cache-hit", flush=True)
    return answer


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(resilient_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

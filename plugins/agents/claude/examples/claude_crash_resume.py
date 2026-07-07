"""Crash and resume: durable Claude agent recovery on Flyte.

Shows that a crash mid-run does not restart the agent from scratch. On the
first attempt the agent does real work (model turns in the Claude Code runtime +
durable tool calls), then the worker is killed (simulated). Flyte retries the
task; on the second attempt:

- the conversation resumes — with ``durable=True`` (the default) ``run_agent``
  mirrors the session transcript to a ``flyte.Checkpoint`` as it runs, and on the
  retry it sets ``resume=<session_id>`` so the Claude runtime continues the prior
  conversation instead of starting over (a deterministic session id derived from
  the task's action keeps every attempt pointed at the same session);
- completed tool calls are cache hits — each tool is a durable Flyte child
  action with ``cache="auto"``, so it isn't re-executed on the retry.

Backend only: per-attempt numbers and the previous attempt's checkpoint are provided
by the platform per attempt — that is where resume is exercised. In ``local`` mode the
crash is skipped and the example just runs once.

Run:  flyte run claude_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os
from pathlib import Path

import flyte
from flyte._image import PythonWheels

from flyteplugins.agents.claude import run_agent, tool

env = flyte.TaskEnvironment(
    "claude-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="anthropic_api_key", as_env_var="ANTHROPIC_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="claude-crash-resume")
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
                package_name="flyteplugins-agents-claude",
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
        model="claude-sonnet-4-5",
        durable=True,  # default — mirror the session to a checkpoint and resume on retry
    )

    # Simulate a worker crash after the agent did real work, but only on the first
    # attempt on a backend. ``FLYTE_ATTEMPT_NUMBER`` is only set per attempt on a
    # backend, so local runs skip the crash and just complete.
    on_backend = os.environ.get("FLYTE_ATTEMPT_NUMBER") is not None
    if on_backend and attempt == 0:
        raise RuntimeError("💥 simulated worker crash (first attempt only)")

    print("✅ completed on retry — tools were cache hits, conversation resumed", flush=True)
    return answer


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(resilient_agent, question="What's the weather and population of Paris?")
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")

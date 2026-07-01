"""Crash-and-resume: durable agent recovery on Flyte.

Shows that a crash mid-run does not redo completed work. On the first attempt
the agent does real work (model turns + tool calls), then the worker is killed
(simulated). Flyte retries the task; on the second attempt the completed model
turns replay from their ``flyte.trace`` records and the tool calls are cache
hits, so the agent finishes without re-calling (or re-billing) the model.

Run this on a Flyte / Union backend, where attempt numbers and durable trace
records are provided per attempt — that is where the replay is visible. In
``local`` mode Flyte still retries, but does not persist trace records across
attempts and does not set a per-attempt number, so this example simply runs once
locally (the crash is skipped) and the replay is not exercised.

Run:  flyte run openai_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os
from pathlib import Path

import flyte
from agents import RunConfig
from agents.models.interface import Model, ModelProvider
from agents.models.multi_provider import MultiProvider
from flyte._image import PythonWheels

from flyteplugins.agents.openai import function_tool, run_agent

env = flyte.TaskEnvironment(
    "openai-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=(
        flyte.Image.from_debian_base(name="openai-crash-resume")
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
                package_name="flyteplugins-agents-openai",
                pre=True,
            ),
        )
    ),
)


# A model wrapper that prints only when the real model is actually called. After
# a crash, replayed turns are served from flyte.trace records and never reach
# this wrapper — so the absence of these lines on the retry is the proof.
class _LoggingModel(Model):
    def __init__(self, inner: Model):
        self._inner = inner

    async def get_response(self, *args, **kwargs):
        # Prints only when the model is ACTUALLY called (a trace miss). On the
        # retry these lines are absent — that absence is the replay.
        print("  🧠 live model call (recorded for replay)", flush=True)
        return await self._inner.get_response(*args, **kwargs)

    def stream_response(self, *args, **kwargs):
        return self._inner.stream_response(*args, **kwargs)

    async def close(self) -> None:
        await self._inner.close()


class LoggingModelProvider(ModelProvider):
    def __init__(self):
        self._inner = MultiProvider()

    def get_model(self, model_name):
        return _LoggingModel(self._inner.get_model(model_name))

    async def aclose(self) -> None:
        await self._inner.aclose()


@function_tool
@env.task(cache="auto", retries=3)
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    print(f"  🛠  get_weather EXECUTED for {city} (cache MISS)", flush=True)
    return f"sunny, 22°C in {city}"


@function_tool
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
        # Inject the logging provider; run_agent wraps it for durability.
        run_config=RunConfig(model_provider=LoggingModelProvider()),
    )

    # Simulate a worker crash after the agent did real work, but only on the
    # first attempt on a backend.
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

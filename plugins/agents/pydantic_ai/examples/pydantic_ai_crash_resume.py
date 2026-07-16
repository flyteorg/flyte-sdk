"""Crash-and-resume: durable Pydantic AI agent recovery on Flyte.

Shows that a crash mid-run does not redo completed work. On the first attempt
the agent does real work (model turns + tool calls), then the worker is killed
(simulated). Flyte retries the task; on the second attempt the completed model
turns replay from their ``flyte.trace`` records and the tool calls are cache
hits, so the agent finishes without re-calling (or re-billing) the model.

How the replay shows up here: ``run_agent`` wraps the model in ``FlyteModel``,
which records every completed ``Model.request`` (one model turn) through the
durable step. We slip a small logging ``Model`` between the real model and that
durable wrapper — it prints only when the model is ACTUALLY called. On the retry
those lines are absent (turns are served from trace records before reaching the
wrapper), and the tools' "EXECUTED ... cache MISS" lines are absent too because
every call is a cache hit.

Run this on a Flyte / Union backend, where attempt numbers and durable trace
records are provided per attempt — that is where the replay is visible. In
``local`` mode Flyte still retries, but does not persist trace records across
attempts and does not set a per-attempt number, so this example simply runs once
locally (the crash is skipped) and the replay is not exercised.

Run:  flyte run pydantic_ai_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os

import flyte
from pydantic_ai.models import Model

from flyteplugins.agents.pydantic_ai import run_agent, tool

env = flyte.TaskEnvironment(
    "pydantic-ai-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="pydantic-ai-crash-resume").with_local_v2_plugins(
        ["flyteplugins-agents-core", "flyteplugins-agents-pydantic-ai"]
    ),
)


# A model wrapper that prints only when the real model is actually called. After
# a crash, replayed turns are served from flyte.trace records and never reach
# this wrapper — so the absence of these lines on the retry is the proof.
class _LoggingModel(Model):
    def __init__(self, inner: Model):
        self._inner = inner

    async def request(self, messages, model_settings, model_request_parameters):
        # Prints only when the model is ACTUALLY called (a trace miss). On the
        # retry these lines are absent — that absence is the replay.
        print("  🧠 live model call (recorded for replay)", flush=True)
        return await self._inner.request(messages, model_settings, model_request_parameters)

    def request_stream(self, *args, **kwargs):
        return self._inner.request_stream(*args, **kwargs)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def system(self) -> str:
        return self._inner.system

    def __getattr__(self, name):
        # Forward everything else (settings, profile, ...) to the real model.
        return getattr(self._inner, name)


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

    # Built lazily inside the task: resolving "openai:gpt-4o" needs the
    # OPENAI_API_KEY secret, which is only present in the task container.
    from pydantic_ai.models import infer_model

    logging_model = _LoggingModel(infer_model("openai:gpt-4o"))

    # run_agent wraps the model in FlyteModel (durable=True is the default), so
    # each turn is: FlyteModel(durable record/replay) -> _LoggingModel -> OpenAI.
    answer = await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        model=logging_model,
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

"""Crash-and-resume: durable LangGraph agent recovery on Flyte.

Shows that a crash mid-run does not redo completed work. On the first attempt
the graph does real work (model turns + tool calls), then the worker is killed
(simulated). Flyte retries the task; on the second attempt the completed model
turns replay from their ``flyte.trace`` records (``ai_node`` records every turn)
and the tool calls are cache hits, so the agent finishes without re-calling (or
re-billing) the model.

Run this on a Flyte / Union backend, where attempt numbers and durable trace
records are provided per attempt — that is where the replay is visible. In
``local`` mode Flyte still retries, but does not persist trace records across
attempts and does not set a per-attempt number, so this example simply runs once
locally (the crash is skipped) and the replay is not exercised.

Run:  flyte run langgraph_crash_resume.py resilient_agent --question "What's the weather and population of Paris?"
      (add `--local` right after `run` to execute locally instead of on the backend)
"""

import os

import flyte
from langchain_core.language_models.chat_models import BaseChatModel

from flyteplugins.agents.langgraph import run_agent, tool

env = flyte.TaskEnvironment(
    "langgraph-crash-resume",
    resources=flyte.Resources(cpu=1),
    secrets=[flyte.Secret(key="openai_api_key", as_env_var="OPENAI_API_KEY")],
    image=flyte.Image.from_debian_base(name="langgraph-crash-resume")
    .with_local_v2_plugins(["flyteplugins-agents-core", "flyteplugins-agents-langgraph"])
    .with_pip_packages("langchain-openai"),
)


# A model wrapper that prints only when the real model is actually called. The
# graph's ``ai_node`` records every turn via flyte.trace, so after a crash the
# replayed turns never reach this wrapper — the absence of these lines on the
# retry is the proof.
class _LoggingChatModel(BaseChatModel):
    inner: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return f"logging-{self.inner._llm_type}"

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Prints only when the model is ACTUALLY called (a trace miss). On the
        # retry these lines are absent — that absence is the replay.
        print("  🧠 live model call (recorded for replay)", flush=True)
        return await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        # Let the inner model format the tools, then re-bind them to this
        # wrapper so generation still routes through the logging override.
        bound = self.inner.bind_tools(tools, **kwargs)
        return self.bind(**dict(getattr(bound, "kwargs", {}) or {}))


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

    from langchain_openai import ChatOpenAI

    answer = await run_agent.aio(
        question,
        tools=[get_weather, get_population],
        instructions="You are a concise city-facts assistant. Use the tools to answer.",
        # The logging wrapper is the chat model; the default graph's ai_node
        # records every turn via flyte.trace (and replays it on retry).
        model=_LoggingChatModel(inner=ChatOpenAI(model="gpt-4o")),
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

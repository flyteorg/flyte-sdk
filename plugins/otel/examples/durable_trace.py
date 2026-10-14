"""A durable agent loop whose trace survives a crash.

The first attempt crashes partway through. The retry resumes: the steps that already completed
are served from the durable log and recorded as spans marked flyte.replayed, alongside the
ones that finally executed. Both attempts share a trace id, so the whole thing reads as one
trace rather than two unrelated ones.

Set OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS to send the trace to a real
backend; without them the spans go to the console, where the replayed steps are still visible.

    python durable_trace.py            # on the cluster in your flyte config

Needs a cluster: the retry is what produces the replay, and retries are a platform feature.
Run with --local and the first attempt's crash is simply the end of it.
"""

import asyncio
import os
import sys

import flyte
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.otel import init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so this runs with or without a backend."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


# disable_batch keeps nothing buffered, so the spans from before a crash are already
# exported when the process dies. Batching is the better default in production.
init(service_name="otel-demo", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheel from ./dist so a cluster run exercises the working
    # tree rather than a PyPI release. Build it with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_demo", image=image)


@flyte.trace
async def think(step: int) -> str:
    """Stands in for a model call. In a real agent this is where an LLM SDK's span would go."""
    await asyncio.sleep(0.2)
    return f"thought-{step}"


@flyte.trace
async def act(thought: str) -> str:
    """Stands in for a tool call."""
    await asyncio.sleep(0.1)
    return f"acted-on-{thought}"


# retries makes the whole story play out in one run: the first attempt crashes, the retry
# replays the steps already in the durable log and finishes.
@env.task(retries=3)
async def agent(steps: int = 5, fail_at: int = 2) -> list[str]:
    # Crash on the first attempt only. FLYTE_ATTEMPT_NUMBER is 1-based, so the first attempt is 1.
    crash = flyte.ctx().attempt_number <= 1

    results = []
    for step in range(steps):
        thought = await think(step)
        results.append(await act(thought))
        if crash and step == fail_at:
            raise RuntimeError(f"crashed at step {step}")
    return results


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    run = runner.run(agent, steps=5)
    print(run.url)
    print("the retry replays the completed steps and finishes, all in one trace")

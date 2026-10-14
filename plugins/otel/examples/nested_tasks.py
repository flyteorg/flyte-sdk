"""Tasks calling tasks, across pods, in one trace.

A child task runs in its own container with its own OpenTelemetry SDK, so nothing in-process
links it to its parent. The plugin publishes the parent's span into custom_context, which
Flyte copies into every sub-action's inputs, so the child picks up the parent on the other
side without anything being passed by hand.

Because inputs are durable, this holds across a resume as well.

    python nested_tasks.py            # on the cluster in your flyte config
    python nested_tasks.py --local    # in-process, no cluster needed
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


init(service_name="otel-nested", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheel from ./dist so a cluster run exercises the working
    # tree rather than a PyPI release. Build it with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_nested", image=image)


@flyte.trace
async def score(item: int) -> int:
    return item * 3


@env.task
async def worker(item: int) -> int:
    return await score(item)


@env.task
async def coordinator(n: int = 3) -> int:
    results = await asyncio.gather(*[worker(item=i) for i in range(n)])
    return sum(results)


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(coordinator, n=3).url)
    print("every worker span sits under the coordinator span, in one trace")

"""The smallest useful setup: every task becomes a span, printed to the console.

Any Flyte task gets a span and any function decorated with flyte.trace
becomes a child span inside it.

    python basic_tracing.py            # on the cluster in your flyte config
    python basic_tracing.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.otel import init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so this runs with or without a backend."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


# Initialized at module scope so the observer is registered before any task starts. The
# task span opens before the task body runs, so init inside a task would miss it.
init(service_name="otel-basic", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheel from ./dist so a cluster run exercises the working
    # tree rather than a PyPI release. Build it with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_basic", image=image)


@flyte.trace
async def double(x: int) -> int:
    return x * 2


@env.task
async def main(n: int = 3) -> int:
    total = 0
    for i in range(n):
        total += await double(i)
    return total


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(main, n=3).url)

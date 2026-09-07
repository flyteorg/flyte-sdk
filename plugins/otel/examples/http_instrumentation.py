"""Third party auto-instrumentation, nested under the task span.

Instrumentation libraries patch a client and emit their own spans. They need no wiring to fit
in: parenting comes from the active context, and the plugin keeps the task span active for
the task body, so every outbound HTTP call lands underneath it.

The same is true of LLM instrumentation; flyteplugins-agento11y builds on this to send
agent generations to Grafana Agent Observability.

Needs an extra dependency:

    pip install opentelemetry-instrumentation-httpx httpx
    python http_instrumentation.py            # on the cluster in your flyte config
    python http_instrumentation.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.otel import init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so this runs with or without a backend."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


image = (
    flyte.Image.from_debian_base()
    .with_pip_packages("opentelemetry-instrumentation-httpx", "httpx")
    # Bake the locally-built plugin wheels from ./dist so a remote run exercises the working
    # tree rather than PyPI releases. Build them with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_httpx", image=image)

init(service_name="otel-httpx", exporter=_console_or_otlp(), disable_batch=True)
HTTPXClientInstrumentor().instrument()


@flyte.trace
async def fetch(url: str) -> int:
    import httpx

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.status_code


@env.task
async def main(url: str = "https://httpbin.org/get") -> int:
    # Three levels: the task span, the traced fetch step, and httpx's own client span.
    return await fetch(url)


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(main, url="https://httpbin.org/get").url)

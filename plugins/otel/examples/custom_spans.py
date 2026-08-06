"""Adding your own spans inside a task.

Spans you create with a plain OpenTelemetry tracer nest inside the task span automatically.
Parenting in OpenTelemetry comes from the active context, and the plugin keeps the task span
active for the whole task body, so there is nothing to extract and no context to pass around.

    python custom_spans.py            # on the cluster in your flyte config
    python custom_spans.py --local    # in-process, no cluster needed
"""

import os
import sys

import flyte
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

from flyteplugins.otel import init


def _console_or_otlp():
    """Console when no OTLP endpoint is set, so this runs with or without a backend."""
    return None if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") else ConsoleSpanExporter()


init(service_name="otel-custom-spans", exporter=_console_or_otlp(), disable_batch=True)

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheel from ./dist so a cluster run exercises the working
    # tree rather than a PyPI release. Build it with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_custom_spans", image=image)

tracer = trace.get_tracer("my.app")


@env.task
async def etl(rows: int = 100) -> int:
    with tracer.start_as_current_span("extract") as span:
        span.set_attribute("rows.requested", rows)
        extracted = rows

    with tracer.start_as_current_span("transform"):
        # Spans nest as deeply as you like; this one lands under transform.
        with tracer.start_as_current_span("validate"):
            transformed = extracted - 1

    with tracer.start_as_current_span("load") as span:
        span.set_attribute("rows.loaded", transformed)

    return transformed


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(etl, rows=100).url)

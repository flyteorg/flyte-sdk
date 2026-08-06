"""Keeping an OpenTelemetry setup you already have.

If your codebase already configures a TracerProvider — its own resource, sampler, exporters,
maybe a vendor SDK — hand it to init instead of letting the plugin build one. Nothing about
your setup changes. The plugin only wraps the provider's id generator, which is what lets
trace ids still be derived from the run so a crash and its resume share one trace.

Passing a provider alongside endpoint, headers, exporter, or resource_attributes is an error
rather than a silent override, because those belong to the provider you configured.

    python existing_provider.py            # on the cluster in your flyte config
    python existing_provider.py --local    # in-process, no cluster needed
"""

import sys

import flyte
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from flyteplugins.otel import init

image = (
    flyte.Image.from_debian_base()
    # Bake the locally-built plugin wheel from ./dist so a cluster run exercises the working
    # tree rather than a PyPI release. Build it with `make dist-all`. flyte itself is not
    # listed: from_debian_base already bakes the local flyte wheel when the installed version
    # is a dev build and ./dist exists, so naming it here would add the same layer twice.
    .with_local_v2_plugins(["flyteplugins-otel"])
)

env = flyte.TaskEnvironment(name="otel_existing_provider", image=image)

# Your own setup, configured at import time exactly as you would without Flyte.
provider = TracerProvider(resource=Resource.create({"service.name": "my-existing-service"}))
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("my.app")

# Adopt it. Sampler, resource, and exporters are left exactly as configured above.
init(tracer_provider=provider)


@env.task
async def main(n: int = 2) -> int:
    with tracer.start_as_current_span("my_work"):
        return n * 2


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte
    print(runner.run(main, n=2).url)

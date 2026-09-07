"""Joining a trace that started outside Flyte.

When a run is kicked off from inside an existing span — a web request, a scheduler, another
service — you usually want the run to appear inside that trace rather than as a separate one.

Inject a W3C carrier into custom_context at submit time and the plugin picks it up: the task
span is started under the caller's span instead of becoming a root.

    python propagate_from_caller.py            # on the cluster in your flyte config
    python propagate_from_caller.py --local    # in-process, no cluster needed
"""

import sys

import flyte
from opentelemetry import trace
from opentelemetry.propagate import inject
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

env = flyte.TaskEnvironment(name="otel_propagate", image=image)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer("my.caller")

init(tracer_provider=trace.get_tracer_provider())


@env.task
async def handle(url: str) -> str:
    return f"handled {url}"


if __name__ == "__main__":
    flyte.init_from_config()
    # init_from_config targets the cluster in your flyte config. --local runs the same
    # task in-process instead; the spans are identical either way.
    runner = flyte.with_runcontext(mode="local") if "--local" in sys.argv else flyte

    with tracer.start_as_current_span("incoming_request") as caller_span:
        carrier: dict[str, str] = {}
        inject(carrier)

        run = flyte.with_runcontext(custom_context=carrier).run(handle, url="https://example.com")

        print(run.url)
        print(f"caller trace: {caller_span.get_span_context().trace_id:032x}")
        print("the run's spans carry that same trace id")

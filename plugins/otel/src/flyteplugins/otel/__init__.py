"""OpenTelemetry tracing for Flyte.

Every task becomes a span, every ``flyte.trace`` step becomes a child span inside it, and
spans created by your own code or by any instrumentation library nest underneath without
wiring. Export goes wherever OTLP goes.

Usage:

    import flyte
    from flyteplugins.otel import init

    # At module scope, not inside a task: the task span opens before the task body runs.
    init(service_name="my-service")

    env = flyte.TaskEnvironment(name="my_env")

    @env.task
    async def main(n: int) -> int:
        ...

With no arguments ``init`` reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS,
so pointing it at a vendor is a matter of setting those usually from a ``flyte.Secret``. If
you already configure OpenTelemetry yourself, pass ``tracer_provider`` and your setup is
adopted unchanged.

Trace context travels in Flyte's ``custom_context`` as a W3C carrier in both directions: a
run submitted inside a caller's span joins that trace, and a child task nests under the task
that spawned it even though it runs in another pod.

Two things are specific to Flyte being durable. When no trace context arrives from outside,
the trace id is derived from the run identity, so the several processes that make up a
crashed-and-resumed run all record into one trace with no coordination. And steps that a
resumed run served from its durable log, which never execute and so would otherwise be
missing, are recorded as spans marked ``flyte.replayed``.
"""

from ._ids import format_trace_id, trace_id_for_run
from ._observer import OtelObserver, RunScopedIdGenerator
from ._setup import get_tracer, init, shutdown

__all__ = [
    "OtelObserver",
    "RunScopedIdGenerator",
    "format_trace_id",
    "get_tracer",
    "init",
    "shutdown",
    "trace_id_for_run",
]

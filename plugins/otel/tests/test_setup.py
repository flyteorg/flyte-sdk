"""Adopting an OpenTelemetry setup that already exists, rather than replacing it."""

import flyte._observe as observe
import pytest
from flyte._observe import TaskInfo, observe_task
from flyte.models import ActionID
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flyteplugins.otel import RunScopedIdGenerator, init, shutdown, trace_id_for_run

ACTION = ActionID(name="a0", run_name="run-abc", project="proj", domain="dev", org="acme")


@pytest.fixture
def clean():
    yield
    shutdown()
    for leftover in list(observe._observers):
        observe.unregister_observer(leftover)


def user_provider():
    """What the Flyte OpenTelemetry docs tell you to set up yourself."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_an_adopted_provider_keeps_exporting_where_it_already_did(clean):
    provider, exporter = user_provider()
    init(tracer_provider=provider)

    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        pass

    assert [span.name for span in exporter.get_finished_spans()] == ["my_task"]


def test_an_adopted_provider_still_gets_run_derived_trace_ids(clean):
    """The durability property must not be something you forfeit by bringing your own setup."""
    provider, exporter = user_provider()
    init(tracer_provider=provider)

    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        pass

    assert exporter.get_finished_spans()[0].context.trace_id == trace_id_for_run(ACTION)
    assert isinstance(provider.id_generator, RunScopedIdGenerator)


def test_adopting_does_not_disturb_the_providers_own_resource(clean):
    provider, _ = user_provider()
    before = provider.resource
    init(tracer_provider=provider)
    assert provider.resource is before


def test_shutdown_does_not_close_a_provider_we_did_not_build(clean):
    """Shutting down an adopted provider would take out tracing the caller still depends on."""
    provider, exporter = user_provider()
    init(tracer_provider=provider)
    shutdown()

    tracer = provider.get_tracer("after-shutdown")
    with tracer.start_as_current_span("still_working"):
        pass

    assert "still_working" in [span.name for span in exporter.get_finished_spans()]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": "https://example.com/otlp"},
        {"headers": {"Authorization": "Basic x"}},
        {"exporter": InMemorySpanExporter()},
        {"resource_attributes": {"deployment.environment": "prod"}},
    ],
)
def test_combining_a_provider_with_exporter_arguments_is_rejected(kwargs, clean):
    """Silently ignoring these would leave someone wondering why their config did nothing."""
    provider, _ = user_provider()
    with pytest.raises(ValueError, match="tracer_provider cannot be combined with"):
        init(tracer_provider=provider, **kwargs)


def test_a_provider_without_an_id_generator_still_records(clean):
    """A stub or no-op provider loses run scoped ids but must not lose tracing."""

    class Minimal:
        def __init__(self):
            self.provider, self.exporter = user_provider()

        def get_tracer(self, *args, **kwargs):
            return self.provider.get_tracer(*args, **kwargs)

    minimal = Minimal()
    init(tracer_provider=minimal)

    with observe_task(TaskInfo(name="my_task", action=ACTION)):
        pass

    assert [span.name for span in minimal.exporter.get_finished_spans()] == ["my_task"]


# --- exporters: OTLP is the default, not a requirement ---


def test_any_span_exporter_works(clean):
    """Nothing here requires OTLP; a plain SpanExporter is enough."""
    exporter = InMemorySpanExporter()
    init(exporter=exporter, disable_batch=True, set_global=False)
    with observe_task(TaskInfo(name="t", action=ACTION)):
        pass
    assert [s.name for s in exporter.get_finished_spans()] == ["t"]


def test_several_exporters_run_side_by_side(clean):
    """A console exporter alongside a real backend is a normal thing to want."""
    first, second = InMemorySpanExporter(), InMemorySpanExporter()
    init(exporter=[first, second], disable_batch=True, set_global=False)
    with observe_task(TaskInfo(name="t", action=ACTION)):
        pass
    assert [s.name for s in first.get_finished_spans()] == ["t"]
    assert [s.name for s in second.get_finished_spans()] == ["t"]


def test_the_otlp_protocol_env_var_is_honoured(clean, monkeypatch):
    """Ignoring it would silently give the wrong transport to a standards-configured backend."""
    from flyteplugins.otel._setup import _resolve_protocol

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
    assert _resolve_protocol(None) == "grpc"

    # The traces-specific variable wins over the general one.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    assert _resolve_protocol(None) == "http/protobuf"

    # An explicit argument wins over both.
    assert _resolve_protocol("grpc") == "grpc"


def test_protocol_defaults_to_http(clean, monkeypatch):
    from flyteplugins.otel._setup import _resolve_protocol

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", raising=False)
    assert _resolve_protocol(None) == "http/protobuf"


def test_an_unknown_protocol_is_rejected(clean):
    """Better than quietly falling back and exporting nowhere."""
    from flyteplugins.otel._setup import _resolve_protocol

    with pytest.raises(ValueError, match="Unsupported OTLP protocol"):
        _resolve_protocol("thrift")


def test_protocol_cannot_be_combined_with_an_adopted_provider(clean):
    provider, _ = user_provider()
    with pytest.raises(ValueError, match="tracer_provider cannot be combined with"):
        init(tracer_provider=provider, protocol="grpc")


def test_concurrent_init_registers_exactly_one_observer(clean):
    """The check-then-set in init must be atomic, or two threads each build a provider.

    The symptom would be duplicate spans and an observer that shutdown cannot fully remove,
    which is exactly the kind of thing that only shows up under load.
    """
    import threading

    import flyte._observe as observe

    barrier = threading.Barrier(8)
    results = []

    def go():
        barrier.wait()  # maximise the overlap on the check
        results.append(init(exporter=InMemorySpanExporter(), disable_batch=True, set_global=False))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(r) for r in results}) == 1, "every caller must get the same observer"
    assert len(observe._observers) == 1, f"expected one registered observer, got {len(observe._observers)}"


def test_no_configured_endpoint_warns_once_instead_of_retrying(clean, monkeypatch):
    """Falling back to the spec default produces a retry storm against localhost:4318.

    That is the driver process's normal state — it imports the module to submit a run and has
    nothing to export — so it has to be one line, not several per batch.
    """
    import logging

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    logging.getLogger("flyte").addHandler(handler)
    try:
        init(service_name="test", set_global=False)
        with observe_task(TaskInfo(name="t", action=ACTION)):
            pass
    finally:
        logging.getLogger("flyte").removeHandler(handler)

    assert sum("no OTLP endpoint configured" in m for m in records) == 1


def test_an_explicit_endpoint_still_builds_an_exporter(clean, monkeypatch):
    """The warning must not become a way to silently stop exporting when it is configured."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.example:4318")
    init(service_name="test", set_global=False)
    from flyteplugins.otel._setup import _state

    assert _state["provider"] is not None
    assert _state["provider"]._active_span_processor._span_processors


@pytest.mark.asyncio
async def test_concurrent_init_from_coroutines(clean):
    """init is sync and holds no await, so coroutines cannot interleave inside it.

    This is why a threading.Lock is right here and an asyncio.Lock would be wrong: the real
    contention is between threads, since Flyte drives sync adapters on a background event loop
    of its own, and an asyncio.Lock guards only within one loop.
    """
    import asyncio

    import flyte._observe as observe

    async def go():
        return init(exporter=InMemorySpanExporter(), disable_batch=True, set_global=False)

    results = await asyncio.gather(*[go() for _ in range(8)])
    assert len({id(r) for r in results}) == 1
    assert len(observe._observers) == 1


def test_concurrent_init_from_threads_each_with_a_loop(clean):
    """The case the lock actually exists for: separate threads, each running their own loop.

    Flyte's sync adapter path (run_agent_sync) dispatches onto a persistent background loop on
    another thread, so two threads really can reach init at once.
    """
    import asyncio
    import threading

    import flyte._observe as observe

    barrier = threading.Barrier(6)
    results = []

    def go():
        async def inner():
            barrier.wait()
            return init(exporter=InMemorySpanExporter(), disable_batch=True, set_global=False)

        results.append(asyncio.run(inner()))

    threads = [threading.Thread(target=go) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(r) for r in results}) == 1, "every caller must get the same observer"
    assert len(observe._observers) == 1

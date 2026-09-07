import flyte._observe as observe
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flyteplugins.otel import init, shutdown


@pytest.fixture
def spans():
    """Initialize the plugin against an in-memory exporter, and tear it down afterwards.

    Yields the exporter, so a test reads recorded spans with ``spans.get_finished_spans()``.
    """
    exporter = InMemorySpanExporter()
    init(exporter=exporter, disable_batch=True, set_global=False, service_name="test")
    try:
        yield exporter
    finally:
        shutdown()
        exporter.clear()
        # A failed test must not leave an observer registered for the next one.
        for leftover in list(observe._observers):
            observe.unregister_observer(leftover)

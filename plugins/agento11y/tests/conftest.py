import flyte._observe as observe
import pytest
from agento11y.exporters.noop import NoopGenerationExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from flyteplugins.agento11y import shutdown


class CapturingExporter(NoopGenerationExporter):
    """Records the generation payloads that would be POSTed to Grafana."""

    def __init__(self):
        self.generations = []

    def export_generations(self, request):
        self.generations.extend(request.generations)
        return super().export_generations(request)


@pytest.fixture
def clean():
    """Tear down all global registration, so one test cannot leak into the next."""
    yield
    shutdown()
    try:
        from flyteplugins.otel import shutdown as otel_shutdown

        otel_shutdown()
    except Exception:
        pass
    for leftover in list(observe._observers):
        observe.unregister_observer(leftover)


@pytest.fixture
def spans():
    return InMemorySpanExporter()


@pytest.fixture
def generations():
    return CapturingExporter()

"""Unit tests for the Flyte tracing processor (best-effort, never raises)."""

from types import SimpleNamespace

from flyteplugins.agents.openai import FlyteTracingProcessor


class _FunctionSpanData:
    type = "function"

    def export(self):
        return {
            "type": "function",
            "name": "get_weather",
            "input": '{"city": "SF"}',
            "output": "sunny",
        }


def _function_span():
    return SimpleNamespace(
        span_data=_FunctionSpanData(),
        error=None,
        started_at="2026-06-16T10:00:00.000Z",
        ended_at="2026-06-16T10:00:00.250Z",
    )


def test_renders_span_without_active_report():
    # No active Flyte report — rendering must be a silent no-op, never raise.
    processor = FlyteTracingProcessor()
    processor.on_trace_start(SimpleNamespace(name="city run"))
    processor.on_span_start(_function_span())
    processor.on_span_end(_function_span())
    processor.force_flush()
    processor.shutdown()


def test_tolerates_malformed_spans():
    processor = FlyteTracingProcessor()
    processor.on_span_end(SimpleNamespace(span_data=None))
    processor.on_span_end(SimpleNamespace())  # missing attributes entirely

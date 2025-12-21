import asyncio
import logging
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from flyte._logging import initialize_logger, logger


class FlyteLogExporter(SpanExporter):
    """
    Custom OpenTelemetry exporter that outputs to Flyte's structured logger.
    Emits metrics compatible with Flyte's JSONFormatter.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            # Calculate duration in seconds
            duration = (span.end_time - span.start_time) / 1_000_000_000

            # Emit with structured logging
            extra = {
                "metric_type": "timer",
                "metric_name": span.name,
                "duration_seconds": duration,
                "trace_id": format(span.context.trace_id, "032x"),
                "span_id": format(span.context.span_id, "016x"),
            }

            logger.info(f"{span.name} completed in {duration:.4f}s", extra=extra)

        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


provider = TracerProvider()
# Use SimpleSpanProcessor for immediate processing (not batched)
processor = SimpleSpanProcessor(FlyteLogExporter())
provider.add_span_processor(processor)

# Sets the global default tracer provider
trace.set_tracer_provider(provider)

# Instrument logging to inject trace context AFTER TracerProvider is set up
from opentelemetry.instrumentation.logging import LoggingInstrumentor

LoggingInstrumentor().instrument(set_logging_format=True)

# Creates a tracer from the global tracer provider
tracer = trace.get_tracer("my.tracer.name")


async def otel_2():
    # what happens if you run two in parallel?
    def dummy_work():
        import time

        time.sleep(0.2)

    async def task1():
        with tracer.start_as_current_span("child") as child:
            logger.info("Processing child task")
            await asyncio.sleep(0.5)

    with tracer.start_as_current_span("parent") as parent:
        logger.info("Starting parent task")
        print("running the sync task")
        dummy_work()

        print("running the async task...")
        await asyncio.gather(task1(), task1())


if __name__ == "__main__":
    initialize_logger(logging.INFO)
    asyncio.run(otel_2())
    # Force flush before exit
    provider.force_flush()

import asyncio
import logging

from opentelemetry.instrumentation.logging import LoggingInstrumentor

LoggingInstrumentor().instrument(set_logging_format=True)

logging.basicConfig(level=logging.INFO)

import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

logger = logging.getLogger("otel.spans")


class OneLineLoggingSpanExporter(SpanExporter):
    def export(self, spans):
        for span in spans:
            ctx = span.get_span_context()
            logger.info(
                "span name=%s duration_ms=%.2f",
                span.name,
                (span.end_time - span.start_time) / 1e6,
            )
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


provider = TracerProvider()
trace.set_tracer_provider(provider)

trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(OneLineLoggingSpanExporter()))

tracer = trace.get_tracer("my.tracer.name")


async def otel_1():
    def dummy_work():
        import time

        time.sleep(0.2)

    async def task1():
        await asyncio.sleep(0.5)

    with tracer.start_as_current_span("parent") as parent:
        print("running the sync task")
        dummy_work()

        tracer.start_span()
        with tracer.start_as_current_span("child") as child:
            print("running the async task...")
            await task1()


async def otel_2():
    # what happens if you run two in parallel?
    def dummy_work():
        import time

        time.sleep(0.2)

    async def task1():
        with tracer.start_as_current_span("child") as child:
            await asyncio.sleep(0.5)

    with tracer.start_as_current_span("parent") as parent:
        print("running the sync task")
        dummy_work()

        print("running the async task...")
        await asyncio.gather(task1(), task1())


if __name__ == "__main__":
    asyncio.run(otel_2())
    # Force flush before exit
    provider.force_flush()
    import time

    time.sleep(0.1)  # Give it time to flush

import os
from contextlib import contextmanager

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

provider = TracerProvider()
if os.getenv("HARNESS_TRACE_CONSOLE") == "1":
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
tracer = provider.get_tracer("scientific-harness", "0.1.0")


@contextmanager
def span(component: str, name: str, episode_id: str = ""):
    kind = component if component in {"AGENT", "RETRIEVER", "TOOL"} else "CHAIN"
    with tracer.start_as_current_span(
        name,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: kind,
            "sciharness.component": component,
            "sciharness.episode_id": episode_id,
        },
    ) as current:
        yield current


def event_context():
    current = trace.get_current_span().get_span_context()
    if current.is_valid:
        return {"trace_id": f"{current.trace_id:032x}", "span_id": f"{current.span_id:016x}"}
    with span("HARNESS", "event.append") as current_span:
        context = current_span.get_span_context()
        return {"trace_id": f"{context.trace_id:032x}", "span_id": f"{context.span_id:016x}"}

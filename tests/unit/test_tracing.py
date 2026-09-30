from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from packages.core.models.schemas import Episode, Event, EventType
from packages.provenance.tracing import provider, span


def test_canonical_trace_links(store):
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ep = Episode(goal="x", domain_pack="demo", agent_id="mock")
    with span("AGENT", "test"):
        event = store.create_episode(
            ep,
            Event(
                episode_id=ep.episode_id,
                branch_id=ep.root_branch_id,
                event_type=EventType.EPISODE_STARTED,
                output={"episode": ep.model_dump(mode="json")},
            ),
        )
    spans = exporter.get_finished_spans()
    assert event.trace_id == f"{spans[-1].context.trace_id:032x}"
    assert event.span_id == f"{spans[-1].context.span_id:016x}"
    assert spans[-1].attributes["openinference.span.kind"] == "AGENT"

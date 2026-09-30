import hashlib
import json

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from packages.core.models.database import EdgeRow, EpisodeRow, EventRow
from packages.core.models.schemas import Episode, Event, EventEdge, EventType
from packages.provenance.tracing import event_context

GENESIS = "0" * 64


def canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def event_hash(event: Event) -> str:
    body = event.model_dump(mode="json", exclude={"state_after_hash"})
    return hashlib.sha256(canonical(body)).hexdigest()


class EventStore:
    """Append-only writer. PostgreSQL serializes writers via the episode row lock."""

    def __init__(self, engine):
        self.engine = engine

    def create_episode(self, episode: Episode, event: Event) -> Event:
        with Session(self.engine) as session, session.begin():
            session.add(
                EpisodeRow(episode_id=episode.episode_id, payload=episode.model_dump(mode="json"))
            )
            session.flush()
            return self._append(session, event)

    def append(self, event: Event, edges: list[EventEdge] | None = None) -> Event:
        with Session(self.engine) as session, session.begin():
            row = session.scalar(
                select(EpisodeRow)
                .where(EpisodeRow.episode_id == event.episode_id)
                .with_for_update()
            )
            if row is None:
                raise KeyError(event.episode_id)
            return self._append(session, event, edges)

    def _append(self, session, event, edges=None):
        if not event.trace_id or not event.span_id:
            event = event.model_copy(update=event_context())
        sequence = (
            session.scalar(
                select(func.count())
                .select_from(EventRow)
                .where(EventRow.episode_id == event.episode_id)
            )
            + 1
        )
        previous = session.scalar(
            select(EventRow)
            .where(EventRow.episode_id == event.episode_id, EventRow.branch_id == event.branch_id)
            .order_by(EventRow.sequence.desc())
            .limit(1)
        )
        parent_id = event.parent_event_id or (previous.event_id if previous else None)
        parent = session.get(EventRow, parent_id) if parent_id else None
        if parent_id and (parent is None or parent.episode_id != event.episode_id):
            raise ValueError("Parent event belongs to another episode or is missing")
        before = (
            previous.payload["state_after_hash"]
            if previous
            else (parent.payload["state_after_hash"] if parent else GENESIS)
        )
        event = event.model_copy(
            update={"sequence": sequence, "parent_event_id": parent_id, "state_before_hash": before}
        )
        event = event.model_copy(update={"state_after_hash": event_hash(event)})
        session.add(
            EventRow(
                event_id=event.event_id,
                episode_id=event.episode_id,
                sequence=sequence,
                branch_id=event.branch_id,
                event_type=event.event_type.value,
                timestamp=event.timestamp,
                payload=event.model_dump(mode="json"),
            )
        )
        session.flush()
        all_edges = list(edges or [])
        if previous:
            all_edges.append(
                EventEdge(
                    source_event_id=previous.event_id,
                    target_event_id=event.event_id,
                    relation="next",
                )
            )
        for edge in all_edges:
            source = session.get(EventRow, edge.source_event_id)
            if (
                source is None
                or source.episode_id != event.episode_id
                or edge.target_event_id != event.event_id
            ):
                raise ValueError("Invalid edge")
            session.add(EdgeRow(**edge.model_dump()))
        return event

    def events(self, episode_id: str) -> list[Event]:
        with Session(self.engine) as session:
            return [
                Event.model_validate(r.payload)
                for r in session.scalars(
                    select(EventRow)
                    .where(EventRow.episode_id == episode_id)
                    .order_by(EventRow.sequence)
                )
            ]

    def episode(self, episode_id: str) -> Episode:
        events = self.events(episode_id)
        if not events:
            raise KeyError(episode_id)
        ep = Episode.model_validate(events[0].output["episode"])
        for event in events:
            if event.event_type in (EventType.EPISODE_COMPLETED, EventType.EPISODE_FAILED):
                ep = ep.model_copy(
                    update={
                        "status": "COMPLETED"
                        if event.event_type == EventType.EPISODE_COMPLETED
                        else "FAILED",
                        "completed_at": event.timestamp,
                    }
                )
        return ep

    def edges(self, episode_id: str) -> list[EventEdge]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(EdgeRow)
                .join(EventRow, EdgeRow.target_event_id == EventRow.event_id)
                .where(EventRow.episode_id == episode_id)
            )
            return [
                EventEdge(
                    source_event_id=r.source_event_id,
                    target_event_id=r.target_event_id,
                    relation=r.relation,
                )
                for r in rows
            ]

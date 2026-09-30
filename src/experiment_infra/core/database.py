from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from packages.core.models.database import Base


class ObjectIndex(Base):
    __tablename__ = "experiment_objects"
    object_id: Mapped[str] = mapped_column(String, primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), index=True)
    kind: Mapped[str] = mapped_column(String)
    created_event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"))


class ProvenanceEdge(Base):
    __tablename__ = "provenance_edges"
    source_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_objects.object_id"), primary_key=True
    )
    target_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_objects.object_id"), primary_key=True
    )
    relation: Mapped[str] = mapped_column(String, primary_key=True)
    created_event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"))

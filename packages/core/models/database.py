from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"
    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))


class EventRow(Base):
    __tablename__ = "events"
    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    episode_id: Mapped[str] = mapped_column(ForeignKey("episodes.episode_id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    branch_id: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    __table_args__ = (UniqueConstraint("episode_id", "sequence"),)


class EdgeRow(Base):
    __tablename__ = "event_edges"
    source_event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), primary_key=True)
    target_event_id: Mapped[str] = mapped_column(ForeignKey("events.event_id"), primary_key=True)
    relation: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, default="1.0")

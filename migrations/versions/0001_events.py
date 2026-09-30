"""Canonical event log and immutable experiment edges."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "episodes",
        sa.Column("episode_id", sa.String(), primary_key=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("episode_id", sa.String(), sa.ForeignKey("episodes.episode_id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("branch_id", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("episode_id", "sequence"),
    )
    for col in ("episode_id", "branch_id", "event_type"):
        op.create_index(f"ix_events_{col}", "events", [col])
    op.create_table(
        "event_edges",
        sa.Column(
            "source_event_id", sa.String(), sa.ForeignKey("events.event_id"), primary_key=True
        ),
        sa.Column(
            "target_event_id", sa.String(), sa.ForeignKey("events.event_id"), primary_key=True
        ),
        sa.Column("relation", sa.String(), primary_key=True),
        sa.Column("schema_version", sa.String(), nullable=False),
    )
    op.execute("""CREATE FUNCTION reject_history_mutation() RETURNS trigger AS $$
    BEGIN RAISE EXCEPTION 'Canonical history is append-only'; END; $$ LANGUAGE plpgsql""")
    for table in ("events", "event_edges"):
        op.execute(
            f"CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE OR TRUNCATE ON {table} FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation()"
        )


def downgrade():
    raise RuntimeError("Destructive history downgrade is deliberately unsupported")

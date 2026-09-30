"""Add immutable object identity and provenance indexes; retain all v1 events."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "experiment_objects",
        sa.Column("object_id", sa.String(), primary_key=True),
        sa.Column(
            "experiment_id", sa.String(), sa.ForeignKey("episodes.episode_id"), nullable=False
        ),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column(
            "created_event_id", sa.String(), sa.ForeignKey("events.event_id"), nullable=False
        ),
    )
    op.create_index("ix_experiment_objects_experiment_id", "experiment_objects", ["experiment_id"])
    op.create_table(
        "provenance_edges",
        sa.Column(
            "source_id",
            sa.String(),
            sa.ForeignKey("experiment_objects.object_id"),
            primary_key=True,
        ),
        sa.Column(
            "target_id",
            sa.String(),
            sa.ForeignKey("experiment_objects.object_id"),
            primary_key=True,
        ),
        sa.Column("relation", sa.String(), primary_key=True),
        sa.Column(
            "created_event_id", sa.String(), sa.ForeignKey("events.event_id"), nullable=False
        ),
    )
    for table in ("experiment_objects", "provenance_edges"):
        op.execute(
            f"CREATE TRIGGER immutable_{table} BEFORE UPDATE OR DELETE OR TRUNCATE ON {table} FOR EACH STATEMENT EXECUTE FUNCTION reject_history_mutation()"
        )


def downgrade():
    raise RuntimeError("Experiment history is persistent; destructive downgrade is unsupported")

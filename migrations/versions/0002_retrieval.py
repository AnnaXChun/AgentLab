"""Rebuildable PostgreSQL FTS + pgvector retrieval index."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""CREATE TABLE retrieval_chunks (
        namespace text NOT NULL,
        evidence_id text NOT NULL,
        schema_version text NOT NULL DEFAULT '1.0',
        document_id text NOT NULL,
        document_hash text NOT NULL,
        source_uri text NOT NULL,
        title text NOT NULL,
        chunk_id text NOT NULL,
        content text NOT NULL,
        embedding vector(64) NOT NULL,
        search tsvector GENERATED ALWAYS AS (to_tsvector('english',content)) STORED,
        PRIMARY KEY(namespace,evidence_id)
    )""")
    op.execute("CREATE INDEX retrieval_fts ON retrieval_chunks USING gin(search)")
    op.execute(
        "CREATE INDEX retrieval_vector ON retrieval_chunks USING hnsw(embedding vector_cosine_ops)"
    )


def downgrade():
    op.drop_table("retrieval_chunks")

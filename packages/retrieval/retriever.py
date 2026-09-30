import hashlib
import json
import math
import re
from typing import Protocol

from sqlalchemy import text

from packages.core.models.schemas import Evidence

DIMENSIONS = 64


class DocumentParser(Protocol):
    def parse(self, data: bytes, mime_type: str) -> str: ...


class TextParser:
    def parse(self, data: bytes, mime_type: str) -> str:
        if mime_type not in ("text/plain", "text/markdown"):
            raise ValueError(
                "V0 parser supports text and Markdown; Docling can implement DocumentParser"
            )
        return data.decode("utf-8")


class EmbeddingProvider(Protocol):
    dimensions: int

    async def embed(self, text: str) -> list[float]: ...


class LocalHashEmbedding:
    """Deterministic feature hashing baseline, not a semantic language model."""

    dimensions = DIMENSIONS

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += (
                1 if digest[4] % 2 else -1
            )
        norm = math.sqrt(sum(x * x for x in vector))
        if not norm:
            vector[0] = norm = 1.0
        return [x / norm for x in vector]


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]: ...


def chunk_document(data: bytes, source_uri: str, title: str, size: int = 1200) -> list[dict]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    content = TextParser().parse(data, "text/markdown")
    digest = hashlib.sha256(data).hexdigest()
    doc_id = hashlib.sha256((source_uri + digest).encode()).hexdigest()
    return [
        {
            "evidence_id": f"evidence:{doc_id}:{index}",
            "document_id": doc_id,
            "document_hash": digest,
            "source_uri": source_uri,
            "title": title,
            "chunk_id": str(index),
            "content": content[start : start + size],
        }
        for index, start in enumerate(range(0, len(content), size))
    ]


class PostgresHybridRetriever:
    def __init__(self, engine, namespace: str, embedding: EmbeddingProvider | None = None):
        self.engine = engine
        self.namespace = namespace
        self.embedding = embedding or LocalHashEmbedding()
        if self.embedding.dimensions != DIMENSIONS:
            raise ValueError("Embedding dimensions require a schema migration")

    async def ingest(self, data: bytes, source_uri: str, title: str) -> list[str]:
        chunks = chunk_document(data, source_uri, title)
        vectors = [await self.embedding.embed(c["content"]) for c in chunks]
        with self.engine.begin() as connection:
            for chunk, vector in zip(chunks, vectors):
                connection.execute(
                    text("""INSERT INTO retrieval_chunks
                    (namespace,evidence_id,document_id,document_hash,source_uri,title,chunk_id,content,embedding)
                    VALUES (:namespace,:evidence_id,:document_id,:document_hash,:source_uri,:title,:chunk_id,:content,CAST(:embedding AS vector))
                    ON CONFLICT (namespace,evidence_id) DO NOTHING"""),
                    {**chunk, "namespace": self.namespace, "embedding": json.dumps(vector)},
                )
        return [c["evidence_id"] for c in chunks]

    async def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not query.strip() or not 1 <= top_k <= 100:
            raise ValueError("Non-empty query and top_k in [1,100] required")
        embedding = await self.embedding.embed(query)
        sql = text("""WITH lexical AS (
          SELECT evidence_id, row_number() OVER (ORDER BY ts_rank_cd(search, plainto_tsquery('english',:query)) DESC,evidence_id) AS rank
          FROM retrieval_chunks WHERE namespace=:namespace AND search @@ plainto_tsquery('english',:query)
          ORDER BY rank LIMIT :pool
        ), semantic AS (
          SELECT evidence_id, row_number() OVER (ORDER BY embedding <=> CAST(:embedding AS vector),evidence_id) AS rank
          FROM retrieval_chunks WHERE namespace=:namespace ORDER BY rank LIMIT :pool
        ), fused AS (
          SELECT coalesce(l.evidence_id,s.evidence_id) AS evidence_id,
            coalesce(1.0/(60+l.rank),0)+coalesce(1.0/(60+s.rank),0) AS score
          FROM lexical l FULL OUTER JOIN semantic s USING(evidence_id)
        ) SELECT c.evidence_id,c.document_id,c.document_hash,c.source_uri,c.title,c.chunk_id,c.content,f.score
          FROM fused f JOIN retrieval_chunks c USING(evidence_id)
          WHERE c.namespace=:namespace ORDER BY f.score DESC,c.evidence_id LIMIT :top_k""")
        with self.engine.connect() as connection:
            rows = (
                connection.execute(
                    sql,
                    {
                        "query": query,
                        "namespace": self.namespace,
                        "embedding": json.dumps(embedding),
                        "pool": max(20, top_k * 4),
                        "top_k": top_k,
                    },
                )
                .mappings()
                .all()
            )
        return [
            Evidence(
                **{k: v for k, v in row.items() if k != "score"},
                retrieval_score=float(row["score"]),
                metadata={"fusion": "rrf", "embedding": "sha256-feature-hash-64"},
            )
            for row in rows
        ]

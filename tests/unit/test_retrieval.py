import pytest

from packages.retrieval.retriever import LocalHashEmbedding, TextParser, chunk_document


async def test_stable_embeddings_and_evidence():
    embed = LocalHashEmbedding()
    assert await embed.embed("candidate A") == await embed.embed("candidate A")
    assert len(await embed.embed("")) == 64
    a = chunk_document(b"hello world", "local://doc", "Doc", size=5)
    assert a == chunk_document(b"hello world", "local://doc", "Doc", size=5)
    assert a[0]["evidence_id"] != chunk_document(b"changed", "local://doc", "Doc")[0]["evidence_id"]
    assert len(a) == 3
    with pytest.raises(ValueError):
        TextParser().parse(b"x", "application/pdf")

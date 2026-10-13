import pytest

from rag_eval.config import Config
from rag_eval.retrieval import build_retriever
from rag_eval.retrieval.embeddings import CachedEmbedder
from rag_eval.retrieval.vector_store import VectorStore


@pytest.fixture
def cfg():
    c = Config(name="unit", pdf_paths=[])
    c.bm25.tokenizer = "whitespace"
    c.retrieval.candidate_k = 10
    return c


def test_vector_store_roundtrip(sample_chunks, fake_embedder):
    store = VectorStore(name="unit test!")
    vecs = fake_embedder.embed([c.text for c in sample_chunks])
    store.upsert(sample_chunks, vecs)
    assert store.count() == 4
    [q] = fake_embedder.embed(["온실가스 배출량 2024년"])
    top_id, sim = store.query(q, k=1)[0]
    assert top_id == "c1" and 0 < sim <= 1.0001


def test_hybrid_retriever_modes(cfg, sample_chunks, fake_embedder):
    r = build_retriever(cfg, sample_chunks, embedder=fake_embedder, store=VectorStore(name="unit"), index=True)
    assert r.search("재생에너지 전환율", k=2, weight_vector=0.0)[0].chunk.page == 25
    assert r.search("재생에너지 전환율", k=2, weight_vector=1.0)[0].chunk.page == 25
    hits = r.search("재생에너지 전환율", k=3, weight_vector=0.5)
    assert hits[0].chunk.page == 25 and [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].vector_score is not None and hits[0].bm25_score is not None


def test_bm25_only_retriever_needs_no_embedder(cfg, sample_chunks):
    r = build_retriever(cfg, sample_chunks, with_vector=False)
    # whitespace tokenizer: "이사회는" != "이사회", so query with exact surface forms
    assert r.search("운영하고 있다", k=1, weight_vector=0.0)[0].chunk.page == 12
    with pytest.raises(RuntimeError):
        r.search("운영하고 있다", k=1, weight_vector=0.5)


def test_cached_embedder_makes_no_second_call(tmp_path, fake_embedder):
    cached = CachedEmbedder(fake_embedder, tmp_path)
    first = cached.embed(["가", "나", "가"])          # duplicate text embedded once
    assert fake_embedder.calls == 1 and cached.misses == 2 and cached.hits == 0
    second = CachedEmbedder(fake_embedder, tmp_path).embed(["나", "가"])   # fresh wrapper, same disk cache
    assert fake_embedder.calls == 1
    assert second == [first[1], first[0]]


def test_build_embedder_rejects_unknown_provider():
    from rag_eval.retrieval.embeddings import build_embedder
    with pytest.raises(ValueError):
        build_embedder("nope", "m", 1)

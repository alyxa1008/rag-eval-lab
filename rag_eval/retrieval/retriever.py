"""Hybrid retriever: vector (ChromaDB) + BM25 (kiwi) fused by weighted sum."""

from __future__ import annotations

from dataclasses import dataclass

from rag_eval.chunking import Chunk
from rag_eval.config import Config
from rag_eval.retrieval.bm25 import BM25Index
from rag_eval.retrieval.embeddings import CachedEmbedder, Embedder, build_embedder
from rag_eval.retrieval.fusion import fuse
from rag_eval.retrieval.tokenizer import build_tokenizer
from rag_eval.retrieval.vector_store import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    rank: int
    score: float
    vector_score: float | None
    bm25_score: float | None


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], bm25: BM25Index,
                 embedder: Embedder | None = None, store: VectorStore | None = None,
                 candidate_k: int = 50):
        self._by_id = {c.id: c for c in chunks}
        self.bm25 = bm25
        self.embedder = embedder
        self.store = store
        self.candidate_k = candidate_k

    @property
    def has_vector(self) -> bool:
        return self.embedder is not None and self.store is not None

    def search(self, query: str, k: int = 10, weight_vector: float = 0.6) -> list[Hit]:
        if weight_vector > 0 and not self.has_vector:
            raise RuntimeError("vector search requested but no embedder/store configured (use weight_vector=0)")
        pool = max(self.candidate_k, k)
        bm25_scores = dict(self.bm25.search(query, pool)) if weight_vector < 1 else {}
        vector_scores: dict[str, float] = {}
        if weight_vector > 0:
            [qvec] = self.embedder.embed([query])
            vector_scores = dict(self.store.query(qvec, pool))
        fused = fuse(vector_scores, bm25_scores, weight_vector)[:k]
        return [
            Hit(chunk=self._by_id[cid], rank=r, score=s,
                vector_score=vector_scores.get(cid), bm25_score=bm25_scores.get(cid))
            for r, (cid, s) in enumerate(fused, start=1)
        ]


def build_retriever(cfg: Config, chunks: list[Chunk], *, with_vector: bool = True,
                    embedder: Embedder | None = None, store: VectorStore | None = None,
                    index: bool = False) -> HybridRetriever:
    """Assemble BM25 (+ vector) retrieval from config. `index=True` (re)embeds and upserts chunks."""
    tokenizer = build_tokenizer(cfg.bm25.tokenizer)
    bm25 = BM25Index([c.id for c in chunks], [c.text for c in chunks], tokenizer, cfg.bm25.k1, cfg.bm25.b)
    if not with_vector:
        return HybridRetriever(chunks, bm25, candidate_k=cfg.retrieval.candidate_k)

    embedder = embedder or CachedEmbedder(
        build_embedder(cfg.embedding.provider, cfg.embedding.model, cfg.embedding.batch_size),
        cfg.embedding.cache_dir)
    store = store or VectorStore(name=cfg.name, path=cfg.retrieval.vector_store_dir)
    if index or store.count() != len(chunks):
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert(chunks, vectors)
    return HybridRetriever(chunks, bm25, embedder, store, candidate_k=cfg.retrieval.candidate_k)

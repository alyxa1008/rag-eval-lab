import json

import pytest

from rag_eval.commands.compare import collect_runs
from rag_eval.config import Config
from rag_eval.retrieval import build_retriever
from rag_eval.retrieval.vector_store import VectorStore


class KeywordReranker:
    """Scores 1.0 for chunks containing the keyword, else 0.0."""
    model = "fake-keyword"

    def __init__(self, keyword):
        self.keyword = keyword

    def score(self, query, texts):
        return [1.0 if self.keyword in t else 0.0 for t in texts]


def test_reranker_reorders_candidates(sample_chunks, fake_embedder):
    cfg = Config(name="unit", pdf_paths=[])
    cfg.bm25.tokenizer = "whitespace"
    r = build_retriever(cfg, sample_chunks, embedder=fake_embedder, store=VectorStore(name="rerank_unit"),
                        reranker=KeywordReranker("이사회"), index=True)
    plain = r.search("재생에너지 전환율", k=2, weight_vector=0.0)
    assert plain[0].chunk.page == 25 and plain[0].rerank_score is None
    # BM25 pool is only chunks with matching terms; use vector so the 이사회 chunk is in the pool
    reranked = r.search("재생에너지 전환율", k=2, weight_vector=1.0, rerank=True)
    assert reranked[0].chunk.page == 12 and reranked[0].rerank_score == 1.0
    assert [h.rank for h in reranked] == [1, 2]


def test_rerank_without_reranker_raises(sample_chunks):
    cfg = Config(name="unit", pdf_paths=[])
    cfg.bm25.tokenizer = "whitespace"
    r = build_retriever(cfg, sample_chunks, with_vector=False)
    with pytest.raises(RuntimeError):
        r.search("x", k=1, weight_vector=0.0, rerank=True)


def test_collect_runs(tmp_path):
    for cfg_name, run, mrr in (("base", "w0.6", 0.95), ("chunk400", "w0.6_rerank", 0.90)):
        d = tmp_path / cfg_name / "dev" / run
        d.mkdir(parents=True)
        (d / "summary.csv").write_text("group,n,recall@1,recall@3,recall@5,mrr\n"
                                       f"all,30,0.9,1.0,1.0,{mrr}\ntype=text,10,1.0,1.0,1.0,1.0\n", encoding="utf-8")
        (d / "run_info.json").write_text(json.dumps({"weight_vector": 0.6, "rerank": "m" if "rerank" in run else None,
                                                     "chunking": {"max_chars": 400 if cfg_name == "chunk400" else 800}}),
                                         encoding="utf-8")
    rows = collect_runs(tmp_path, "dev")
    assert [(r["config"], r["run"], r["chunk"], r["rerank"], r["mrr"]) for r in rows] == [
        ("base", "w0.6", 800, "off", 0.95), ("chunk400", "w0.6_rerank", 400, "on", 0.90)]
    assert collect_runs(tmp_path, "heldout") == []

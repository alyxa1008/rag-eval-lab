"""Run every eval question through a retriever and collect page-level results."""

from __future__ import annotations

from rag_eval.evalset import EvalItem
from rag_eval.evaluation.metrics import QueryResult
from rag_eval.retrieval.retriever import HybridRetriever


def evaluate(retriever: HybridRetriever, items: list[EvalItem], top_k: int, weight_vector: float) -> list[QueryResult]:
    results: list[QueryResult] = []
    for it in items:
        hits = retriever.search(it.question, k=top_k, weight_vector=weight_vector)
        results.append(QueryResult(
            id=it.id, question=it.question, type=it.type, split=it.split, gold_pages=it.gold_pages,
            top_pages=[h.chunk.page for h in hits],
            top_chunk_ids=[h.chunk.id for h in hits],
            top_scores=[round(h.score, 4) for h in hits],
        ))
    return results

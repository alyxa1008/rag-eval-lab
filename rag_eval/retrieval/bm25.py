"""BM25 over chunk texts."""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from rag_eval.retrieval.tokenizer import Tokenizer


class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], tokenizer: Tokenizer, k1: float = 1.5, b: float = 0.75):
        if len(ids) != len(texts):
            raise ValueError("ids and texts must align")
        self.ids = list(ids)
        self._tok = tokenizer
        corpus = [tokenizer.tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self._bm25.get_scores(self._tok.tokenize(query)), dtype=float)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]

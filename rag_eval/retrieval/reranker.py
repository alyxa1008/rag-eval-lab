"""Cross-encoder reranking of fused candidates (BAAI/bge-reranker-v2-m3 by default)."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    model: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str, batch_size: int = 16, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = model
        self.batch_size = batch_size
        self._ce = CrossEncoder(model, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._ce.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

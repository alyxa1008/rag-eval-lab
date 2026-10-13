"""Score fusion: min-max normalise each retriever's candidates, then w*vector + (1-w)*bm25."""

from __future__ import annotations


def min_max(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def fuse(vector: dict[str, float], bm25: dict[str, float], weight_vector: float) -> list[tuple[str, float]]:
    """Return (id, fused score) sorted descending. Ids missing from one side get 0 there."""
    if not 0.0 <= weight_vector <= 1.0:
        raise ValueError("weight_vector must be in [0, 1]")
    v, b = min_max(vector), min_max(bm25)
    ids = set(v) | set(b)
    fused = {i: weight_vector * v.get(i, 0.0) + (1.0 - weight_vector) * b.get(i, 0.0) for i in ids}
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))

"""Bootstrap confidence intervals over questions (paired across runs)."""

from __future__ import annotations

import numpy as np


def _metrics(ranks: np.ndarray, k: int) -> tuple[float, float]:
    """(Recall@k, MRR) for an array of first-hit ranks (0 = miss)."""
    hit = (ranks > 0) & (ranks <= k)
    rr = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    return float(hit.mean()), float(rr.mean())


def bootstrap_ci(ranks: list[int], k: int = 1, n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    """Return {'recall': (point, lo, hi), 'mrr': (point, lo, hi)} for one run."""
    r = np.asarray(ranks, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    samples = np.array([_metrics(r[i], k) for i in idx])
    point = _metrics(r, k)
    lo = np.quantile(samples, alpha / 2, axis=0)
    hi = np.quantile(samples, 1 - alpha / 2, axis=0)
    return {"recall": (point[0], float(lo[0]), float(hi[0])), "mrr": (point[1], float(lo[1]), float(hi[1]))}


def paired_diff_ci(ranks_a: list[int], ranks_b: list[int], k: int = 1, n_boot: int = 5000, seed: int = 42,
                   alpha: float = 0.05):
    """CI of (B - A) resampling the same questions for both runs. Also returns P(B - A > 0)."""
    a, b = np.asarray(ranks_a, dtype=float), np.asarray(ranks_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired runs must have the same questions")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = np.array([np.subtract(_metrics(b[i], k), _metrics(a[i], k)) for i in idx])
    point = np.subtract(_metrics(b, k), _metrics(a, k))
    lo = np.quantile(diffs, alpha / 2, axis=0)
    hi = np.quantile(diffs, 1 - alpha / 2, axis=0)
    return {
        "recall": (float(point[0]), float(lo[0]), float(hi[0]), float((diffs[:, 0] > 0).mean())),
        "mrr": (float(point[1]), float(lo[1]), float(hi[1]), float((diffs[:, 1] > 0).mean())),
    }

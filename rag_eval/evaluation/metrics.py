"""Retrieval metrics: Recall@k and MRR at page level, overall and per question type."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    id: str
    question: str
    type: str
    split: str
    gold_pages: list[int]
    top_pages: list[int]            # page of each retrieved chunk, rank order
    top_chunk_ids: list[str] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)

    @property
    def first_hit_rank(self) -> int | None:
        return first_hit_rank(self.top_pages, self.gold_pages)

    def hit_at(self, k: int) -> bool:
        r = self.first_hit_rank
        return r is not None and r <= k

    @property
    def reciprocal_rank(self) -> float:
        r = self.first_hit_rank
        return 0.0 if r is None else 1.0 / r


def first_hit_rank(ranked_pages: list[int], gold_pages: list[int]) -> int | None:
    """1-based rank of the first retrieved chunk whose page is a gold page; None if no hit."""
    gold = set(gold_pages)
    for rank, page in enumerate(ranked_pages, start=1):
        if page in gold:
            return rank
    return None


@dataclass
class GroupSummary:
    group: str
    n: int
    recall: dict[int, float]
    mrr: float

    def as_row(self, ks: tuple[int, ...]) -> dict[str, object]:
        row: dict[str, object] = {"group": self.group, "n": self.n}
        for k in ks:
            row[f"recall@{k}"] = round(self.recall[k], 4)
        row["mrr"] = round(self.mrr, 4)
        return row


def _summarize_group(name: str, results: list[QueryResult], ks: tuple[int, ...]) -> GroupSummary:
    n = len(results)
    recall = {k: (sum(r.hit_at(k) for r in results) / n if n else 0.0) for k in ks}
    mrr = sum(r.reciprocal_rank for r in results) / n if n else 0.0
    return GroupSummary(group=name, n=n, recall=recall, mrr=mrr)


def summarize(results: list[QueryResult], ks: tuple[int, ...] = (1, 3, 5)) -> list[GroupSummary]:
    """Overall summary first, then one per question type (sorted by type name)."""
    out = [_summarize_group("all", results, ks)]
    by_type: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_type[r.type].append(r)
    for t in sorted(by_type):
        out.append(_summarize_group(f"type={t}", by_type[t], ks))
    return out

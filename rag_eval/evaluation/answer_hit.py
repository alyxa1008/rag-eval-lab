"""Chunk-level check: does a retrieved chunk on a gold page actually contain the answer?

Page-level Recall rewards big chunks (a chunk that covers the whole page always
"hits"). AnsHit@k is stricter: the chunk must be on a gold page AND contain at least
one answer key. Keys are derived from the answer string: the numbers of 2+ characters
if there are any (units like "톤CO2e" are ignored), otherwise Korean/Latin tokens of
3+ characters.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z가-힣]+")
_STOP = {"이상", "이내", "수준", "대비", "기준", "각각", "미만", "이하", "합계"}


def extract_keys(answer: str) -> list[str]:
    nums = [n for n in _NUM.findall(answer) if len(n) >= 2]
    if nums:
        return list(dict.fromkeys(nums))
    words = [w for w in _WORD.findall(answer) if len(w) >= 3 and w not in _STOP]
    return list(dict.fromkeys(words))


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def chunk_contains_answer(chunk_text: str, keys: list[str]) -> bool:
    t = _norm(chunk_text)
    return any(_norm(k) in t for k in keys)


def answer_hit_rank(top_pages: list[int], top_texts: list[str], gold_pages: list[int], answer: str) -> int | None:
    """1-based rank of the first chunk that is on a gold page and contains an answer key."""
    keys = extract_keys(answer)
    gold = set(gold_pages)
    for rank, (page, text) in enumerate(zip(top_pages, top_texts), start=1):
        if page in gold and chunk_contains_answer(text, keys):
            return rank
    return None

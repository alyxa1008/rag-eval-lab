import hashlib
import math

import pytest

from rag_eval.chunking import Chunk


class FakeEmbedder:
    """Deterministic, offline embedding: hashed character bigrams -> unit vector."""

    model = "fake-bigram"
    dim = 64

    def __init__(self):
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        t = text.lower().replace(" ", "")
        for i in range(len(t) - 1):
            h = int(hashlib.md5(t[i:i + 2].encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]


def make_chunk(cid: str, text: str, page: int, type_: str = "text") -> Chunk:
    return Chunk(id=cid, text=text, page=page, type=type_, source="doc")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def sample_chunks():
    return [
        make_chunk("c1", "2024년 온실가스 배출량은 14,889천 톤CO2e로 전년 대비 증가했다.", 68),
        make_chunk("c2", "임직원 수는 262,647명이며 국내 125,297명, 해외 137,350명이다.", 64),
        make_chunk("c3", "재생에너지 전환율은 2024년 기준 24.8%를 달성했다.", 25),
        make_chunk("c4", "이사회는 지속가능경영위원회를 운영하고 있다.", 12),
    ]

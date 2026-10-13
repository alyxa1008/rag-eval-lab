import pytest

from rag_eval.evaluation.answer_hit import answer_hit_rank, chunk_contains_answer, extract_keys
from rag_eval.evaluation.bootstrap import bootstrap_ci, paired_diff_ci


def test_extract_keys():
    assert extract_keys("14,889천 톤CO2e") == ["14,889"]          # numbers only when present
    assert extract_keys("사내이사 3인, 사외이사 6인") == ["사내이사", "사외이사"]
    assert extract_keys("90일 이내") == ["90"]
    assert extract_keys("열여덟 번째, 안진회계법인(ISAE 3000 기준)") == ["3000"]
    assert extract_keys("Bespoke AI 스팀 로봇청소기") == ["Bespoke", "로봇청소기"]


def test_chunk_contains_answer_ignores_whitespace():
    assert chunk_contains_answer("매출액 | 조 원 | 302.2 | 258.9 | 300.9", extract_keys("300.9조 원"))
    assert chunk_contains_answer("사내이사 3인 (전영현, 노태문, 송재혁)과 사외이사 6인", extract_keys("사내이사 3인, 사외이사 6인"))
    assert not chunk_contains_answer("매출액 | 조 원 | 302.2", extract_keys("300.9조 원"))


def test_numeric_keys_match_whole_tokens_only():
    assert not chunk_contains_answer("사원 % 43.9 42.5 41.9", extract_keys("1.9%"))
    assert chunk_contains_answer("장애인 고용률 % 1.6 1.8 1.9", extract_keys("1.9%"))
    assert not chunk_contains_answer("장애인 임직원 수 명 1,998", extract_keys("98%"))
    assert chunk_contains_answer("폐기물 재활용률 % 97 97 98", extract_keys("98%"))


def test_answer_hit_rank_requires_gold_page_and_key():
    pages = [62, 5, 62]
    texts = ["매출 소개 문단", "매출 300.9조 원", "매출액 302.2 258.9 300.9"]
    assert answer_hit_rank(pages, texts, [62], "300.9조 원") == 3      # rank 2 has the key but wrong page
    assert answer_hit_rank(pages, texts, [99], "300.9조 원") is None


def test_bootstrap_ci_brackets_point_estimate():
    ranks = [1, 1, 2, 0, 1, 3, 1, 1, 0, 1]
    ci = bootstrap_ci(ranks, k=1, n_boot=500)
    r, lo, hi = ci["recall"]
    assert r == pytest.approx(0.6) and lo <= r <= hi and 0.0 <= lo and hi <= 1.0
    same = bootstrap_ci([1] * 10, k=1, n_boot=200)
    assert same["recall"] == (1.0, 1.0, 1.0)


def test_paired_diff_ci_sign_and_identity():
    a = [1, 2, 1, 0, 3, 1]
    b = [1, 1, 1, 1, 1, 1]
    d = paired_diff_ci(a, b, k=1, n_boot=500)
    assert d["recall"][0] == pytest.approx(0.5) and d["recall"][1] > 0        # b clearly better
    zero = paired_diff_ci(a, a, k=1, n_boot=200)
    assert zero["mrr"] == (0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2], [1], k=1)

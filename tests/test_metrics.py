import pytest

from rag_eval.evaluation.metrics import QueryResult, first_hit_rank, summarize


def qr(id_, type_, gold, top):
    return QueryResult(id=id_, question="q", type=type_, split="dev", gold_pages=gold, top_pages=top)


def test_first_hit_rank():
    assert first_hit_rank([5, 12, 13], [12, 13]) == 2
    assert first_hit_rank([12], [12]) == 1
    assert first_hit_rank([1, 2, 3], [9]) is None
    assert first_hit_rank([], [9]) is None


def test_query_result_hit_and_rr():
    r = qr("q1", "number", [7], [3, 3, 7, 9])
    assert r.first_hit_rank == 3
    assert (r.hit_at(1), r.hit_at(3), r.hit_at(5)) == (False, True, True)
    assert r.reciprocal_rank == pytest.approx(1 / 3)
    miss = qr("q2", "text", [1], [2, 3])
    assert miss.reciprocal_rank == 0.0 and not miss.hit_at(5)


def test_summarize_overall_and_by_type():
    results = [
        qr("q1", "number", [7], [7, 1, 2]),      # rank 1
        qr("q2", "number", [7], [1, 2, 7]),      # rank 3
        qr("q3", "text", [4], [1, 2, 3]),        # miss
        qr("q4", "table", [9], [1, 9]),          # rank 2
    ]
    s = summarize(results, ks=(1, 3, 5))
    assert [g.group for g in s] == ["all", "type=number", "type=table", "type=text"]
    overall = s[0]
    assert overall.n == 4
    assert overall.recall == {1: 0.25, 3: 0.75, 5: 0.75}
    assert overall.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)
    number = s[1]
    assert number.n == 2 and number.recall[1] == 0.5 and number.recall[3] == 1.0
    assert s[0].as_row((1, 3, 5)) == {"group": "all", "n": 4, "recall@1": 0.25, "recall@3": 0.75, "recall@5": 0.75,
                                      "mrr": round(overall.mrr, 4)}


def test_summarize_empty():
    assert summarize([], ks=(1,))[0].n == 0

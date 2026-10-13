import pytest

from rag_eval.retrieval.fusion import fuse, min_max


def test_min_max_scales_to_unit_interval():
    assert min_max({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}
    assert min_max({}) == {}
    assert min_max({"a": 3.0}) == {"a": 1.0}           # constant scores -> all 1
    assert min_max({"a": 1.0, "b": 1.0}) == {"a": 1.0, "b": 1.0}


def test_fuse_weight_extremes_follow_single_retriever():
    vec = {"a": 0.9, "b": 0.5, "c": 0.1}
    bm = {"c": 10.0, "b": 5.0, "d": 2.0, "e": 1.0}
    assert [i for i, _ in fuse(vec, bm, 1.0)][:3] == ["a", "b", "c"]
    assert [i for i, _ in fuse(vec, bm, 0.0)][:3] == ["c", "b", "d"]
    # the lowest-scored retrieved id normalises to 0, tying with ids never retrieved
    assert dict(fuse(vec, bm, 0.0))["e"] == 0.0


def test_fuse_missing_ids_count_as_zero_and_weighted_sum_is_exact():
    vec = {"a": 1.0, "b": 0.0}
    bm = {"b": 1.0, "c": 0.0}
    fused = dict(fuse(vec, bm, 0.6))
    assert fused["a"] == pytest.approx(0.6)   # vector only
    assert fused["b"] == pytest.approx(0.4)   # bm25 only
    assert fused["c"] == pytest.approx(0.0)


def test_fuse_rejects_bad_weight():
    with pytest.raises(ValueError):
        fuse({}, {}, 1.5)

from rag_eval.retrieval.bm25 import BM25Index
from rag_eval.retrieval.tokenizer import KiwiTokenizer, WhitespaceTokenizer


def test_whitespace_tokenizer_lowercases_and_splits_punctuation():
    assert WhitespaceTokenizer().tokenize("Scope 1, 배출량(tCO2e)!") == ["scope", "1", "배출량", "tco2e"]


def test_bm25_ranks_matching_doc_first(sample_chunks):
    idx = BM25Index([c.id for c in sample_chunks], [c.text for c in sample_chunks], WhitespaceTokenizer())
    hits = idx.search("재생에너지 전환율", k=3)
    assert hits[0][0] == "c3"
    assert all(score > 0 for _, score in hits)
    assert idx.search("존재하지않는단어", k=3) == []


def test_kiwi_tokenizer_strips_particles_and_keeps_numbers():
    toks = KiwiTokenizer().tokenize("2024년 삼성전자의 온실가스 배출량은 4,725천 톤이었다.")
    assert "삼성전자" in toks and "배출량" in toks and "4,725" in toks
    assert "의" not in toks and "은" not in toks


def test_kiwi_bm25_matches_inflected_query(sample_chunks):
    idx = BM25Index([c.id for c in sample_chunks], [c.text for c in sample_chunks], KiwiTokenizer())
    # "임직원이 몇 명" vs doc "임직원 수는 262,647명" — whitespace tokens would not match "임직원이"
    assert idx.search("임직원이 몇 명인가요", k=1)[0][0] == "c2"

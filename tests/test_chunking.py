from rag_eval.chunking import chunk_pages, split_text
from rag_eval.config import ChunkingConfig
from rag_eval.parsing.models import Page, Table


def test_split_text_short_returns_single_piece():
    assert split_text("짧은 문장", 100) == ["짧은 문장"]


def test_split_text_empty():
    assert split_text("   ", 100) == []


def test_split_text_respects_max_and_covers_everything():
    text = " ".join(f"단어{i}" for i in range(300))
    pieces = split_text(text, max_chars=120, overlap_chars=20)
    assert all(len(p) <= 120 for p in pieces)
    assert pieces[0].startswith("단어0")
    assert pieces[-1].endswith("단어299")
    assert len(pieces) > 5


def test_split_text_prefers_paragraph_breaks():
    text = "가" * 60 + "\n\n" + "나" * 60
    pieces = split_text(text, max_chars=100)
    assert pieces == ["가" * 60, "나" * 60]


def test_split_text_without_breaks_still_progresses():
    text = "가" * 1000
    pieces = split_text(text, max_chars=300, overlap_chars=50)
    assert "".join(p[:250] if i < len(pieces) - 1 else p for i, p in enumerate(pieces))  # no infinite loop
    assert all(len(p) <= 300 for p in pieces)


def test_chunk_pages_keeps_page_metadata_and_splits_tables():
    table = Table(page=2, index=0, header=["항목", "2023", "2024"],
                  rows=[[f"지표{i}", str(i), str(i * 2)] for i in range(25)])
    pages = [
        Page(page=1, text="첫 페이지 본문입니다."),
        Page(page=2, text="두 번째 페이지 본문", tables=[table]),
    ]
    cfg = ChunkingConfig(max_chars=500, overlap_chars=0, table_rows_per_chunk=10)
    chunks = chunk_pages(pages, cfg, source="doc")

    assert all(c.page in (1, 2) for c in chunks)
    text_chunks = [c for c in chunks if c.type == "text"]
    table_chunks = [c for c in chunks if c.type == "table"]
    assert [c.page for c in text_chunks] == [1, 2]
    assert len(table_chunks) == 3                     # 25 rows -> 10 + 10 + 5
    assert [c.meta["row_end"] - c.meta["row_start"] for c in table_chunks] == [10, 10, 5]
    assert all(c.page == 2 for c in table_chunks)
    assert "항목: 지표0 | 2023: 0 | 2024: 0" in table_chunks[0].text
    assert table_chunks[0].id == "doc-p002-tb1-1"
    assert len({c.id for c in chunks}) == len(chunks)  # ids unique

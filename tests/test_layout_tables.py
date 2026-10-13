from rag_eval.parsing.layout_tables import (
    LayoutConfig, clean_label, extract_layout_tables_from_words, split_cells, split_segments,
)

CW = 6.0  # fake character width


def W(text: str, x0: float, y: float):
    """Make a word box; consecutive words in one cell are placed 3pt apart."""
    return (x0, y, x0 + CW * len(text), y + 8, text)


def line(y: float, *cells: tuple[float, str]):
    """cells: (x0, 'text with spaces') -> words 3pt apart inside a cell."""
    words = []
    for x0, text in cells:
        x = x0
        for w in text.split():
            words.append(W(w, x, y))
            x += CW * len(w) + 3
    return words


def test_clean_label_footnotes():
    assert clean_label(["직접", "배출량(Scope", "1)"]) == "직접 배출량(Scope 1)"
    assert clean_label(["온실가스", "관리(Scope", "1,", "2)", "1),2)"]) == "온실가스 관리(Scope 1, 2)"
    assert clean_label(["총", "임직원1)"]) == "총 임직원"
    assert clean_label(["배출량", "집약도", "3),", "4)"]) == "배출량 집약도"
    assert clean_label(["(Tier1)", "협력회사"]) == "(Tier1) 협력회사"


def test_split_cells_merges_close_words():
    cells = split_cells(line(0, (40, "조 원"), (120, "258.9")), cell_gap=8)
    assert [c[0] for c in cells] == ["조 원", "258.9"]


def test_split_segments_on_value_run_and_repeated_year():
    cells = [("매출액", 0, 30), ("조 원", 50, 70), ("1", 100, 105), ("2", 120, 125), ("자본재", 200, 230), ("3", 300, 305), ("4", 320, 325)]
    segs = split_segments(cells, min_values=2)
    assert [s.cells for s in segs] == [["매출액", "조 원", "1", "2"], ["자본재", "3", "4"]]
    header = [("A", 0, 10), ("2023년", 50, 80), ("2024년", 100, 130), ("2023년", 300, 330), ("2024년", 350, 380)]
    assert [s.cells for s in split_segments(header, 2)] == [["A", "2023년", "2024년"], ["2023년", "2024년"]]


def test_single_table_with_year_header():
    words = (
        line(10, (40, "핵심 재무 성과"), (250, "2023년"), (320, "2024년"))
        + line(25, (40, "매출액"), (180, "조 원"), (250, "258.9"), (320, "300.9"))
        + line(40, (40, "영업이익 1)"), (180, "조 원"), (250, "6.6"), (320, "32.7"))
    )
    tables = extract_layout_tables_from_words(words, page=62)
    assert len(tables) == 1
    t = tables[0]
    assert t.kind == "layout" and t.page == 62 and t.title == "핵심 재무 성과"
    assert t.header == ["항목", "단위", "2023년", "2024년"]
    assert t.rows == [["매출액", "조 원", "258.9", "300.9"], ["영업이익", "조 원", "6.6", "32.7"]]


def test_side_by_side_tables_and_sections():
    words = (
        line(10, (40, "임직원 현황"), (250, "2023년"), (320, "2024년"), (650, "2023년"), (720, "2024년"))
        + line(25, (40, "총 임직원1)"), (180, "명"), (250, "267,860"), (320, "262,647"), (430, "[지역별 근로자 수]"))
        + line(40, (45, "해외"), (180, "명"), (250, "147,104"), (320, "137,350"), (435, "한국"), (570, "명"), (650, "43,902"), (720, "42,589"))
        + line(55, (40, "[연령층별]"), (435, "유럽"), (570, "명"), (650, "5,479"), (720, "5,115"))
        + line(70, (45, "30대 미만"), (180, "명"), (250, "72,525"), (320, "63,531"), (430, "1) 각주 설명"))
        + line(85, (40, "[직급별]"), (430, "퇴직률"), (650, "2023년"), (720, "2024년"))
        + line(100, (45, "사원"), (180, "명"), (250, "174,060"), (320, "164,895"), (435, "전체"), (570, "%"), (650, "10.6"), (720, "10.1"))
    )
    tables = extract_layout_tables_from_words(words, page=64)
    assert [(t.title, t.n_rows) for t in tables] == [("임직원 현황", 4), ("", 2), ("퇴직률", 1)]
    left, right, right2 = tables
    assert left.rows[0] == ["총 임직원", "명", "267,860", "262,647"]
    assert left.rows[2] == ["[연령층별] 30대 미만", "명", "72,525", "63,531"]
    assert left.rows[3] == ["[직급별] 사원", "명", "174,060", "164,895"]
    assert right.rows == [["[지역별 근로자 수] 한국", "명", "43,902", "42,589"],
                          ["[지역별 근로자 수] 유럽", "명", "5,479", "5,115"]]
    assert right2.rows == [["전체", "%", "10.6", "10.1"]]
    assert [t.index for t in tables] == [0, 1, 2]


def test_noise_without_year_header_is_dropped():
    words = line(10, (40, "본문 문장입니다"), (300, "93.1%"), (360, "93.4%")) + line(25, (40, "다른 문장"), (300, "1"), (360, "2"))
    assert extract_layout_tables_from_words(words, page=14) == []
    # three or more rows without a header are kept, with generic column names
    words += line(40, (40, "셋째 줄"), (300, "3"), (360, "4"))
    tables = extract_layout_tables_from_words(words, page=14)
    assert len(tables) == 1 and tables[0].header == ["항목", "단위", "col3", "col4"]


def test_min_values_config():
    words = line(10, (40, "A"), (250, "2023년"), (320, "2024년"), (390, "2025년")) + line(25, (40, "x"), (180, "명"), (250, "1"), (320, "2"))
    assert extract_layout_tables_from_words(words, page=1, cfg=LayoutConfig(min_values=3)) == []
    assert extract_layout_tables_from_words(words, page=1, cfg=LayoutConfig(min_values=2))[0].n_rows == 1

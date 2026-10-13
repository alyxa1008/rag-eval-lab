from rag_eval.parsing.tables import clean_table, row_to_text, rows_to_text


def test_clean_table_strips_cells_and_drops_empty_rows():
    raw = [
        ["항목", "단위", "2024"],
        [None, None, None],
        ["온실가스\n배출량", "tCO2eq", " 1,234 "],
        ["", "", ""],
    ]
    t = clean_table(raw, page=5, index=0)
    assert t is not None
    assert t.header == ["항목", "단위", "2024"]
    assert t.rows == [["온실가스 배출량", "tCO2eq", "1,234"]]
    assert t.page == 5 and t.n_rows == 1


def test_clean_table_fills_blank_header_and_ragged_rows():
    raw = [["", "2023"], ["매출", "100", "extra"]]
    t = clean_table(raw, page=1, index=0)
    assert t is not None
    assert t.header == ["col1", "2023", "col3"]
    assert t.rows == [["매출", "100", "extra"]]


def test_clean_table_rejects_noise():
    assert clean_table([["only header"]], page=1, index=0) is None
    assert clean_table([["a"], ["b"], ["c"]], page=1, index=0) is None   # one column


def test_row_to_text_skips_empty_cells():
    assert row_to_text(["항목", "2023", "2024"], ["매출", "", "200"]) == "항목: 매출 | 2024: 200"
    assert rows_to_text(["a", "b"], [["1", "2"], ["3", "4"]]) == "a: 1 | b: 2\na: 3 | b: 4"

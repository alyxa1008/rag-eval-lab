import pytest

from rag_eval.evalset import load_eval_set, parse_gold_pages, validate_rows

GOOD = {"id": "q001", "question": "2024년 온실가스 배출량은?", "answer": "1,234 tCO2eq",
        "gold_pages": "12;13", "type": "number", "split": "dev"}


def test_parse_gold_pages():
    assert parse_gold_pages("12;13") == [12, 13]
    assert parse_gold_pages(" 7 ") == [7]
    assert parse_gold_pages("") == []


def test_validate_good_row():
    assert validate_rows([GOOD]) == []


@pytest.mark.parametrize("field,value,fragment", [
    ("id", "1", "id must look like"),
    ("question", " ", "question is empty"),
    ("answer", "", "answer is empty"),
    ("gold_pages", "", "gold_pages is empty"),
    ("gold_pages", "12,13", "integers separated"),
    ("gold_pages", "0", "must be positive"),
    ("type", "numeric", "type must be"),
    ("split", "test", "split must be"),
])
def test_validate_bad_field(field, value, fragment):
    errors = validate_rows([{**GOOD, field: value}])
    assert any(fragment in e for e in errors), errors


def test_validate_duplicate_ids():
    errors = validate_rows([GOOD, dict(GOOD)])
    assert any("duplicate id q001" in e for e in errors)


def test_load_eval_set_filters_split(tmp_path):
    p = tmp_path / "eval.csv"
    p.write_text(
        "id,question,answer,gold_pages,type,split\n"
        "q001,질문1,답1,3,text,dev\n"
        "q002,질문2,답2,4;5,multi,heldout\n",
        encoding="utf-8",
    )
    assert [it.id for it in load_eval_set(p)] == ["q001", "q002"]
    assert [it.id for it in load_eval_set(p, split="dev")] == ["q001"]
    assert load_eval_set(p, split="heldout")[0].gold_pages == [4, 5]


def test_load_eval_set_rejects_bad_header(tmp_path):
    p = tmp_path / "eval.csv"
    p.write_text("id,question\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_eval_set(p)

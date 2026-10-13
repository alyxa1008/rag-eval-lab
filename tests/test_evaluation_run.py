import csv

from rag_eval.config import Config
from rag_eval.evalset import EvalItem
from rag_eval.evaluation.metrics import summarize
from rag_eval.evaluation.report import markdown_table, missed_questions, write_results
from rag_eval.evaluation.runner import evaluate
from rag_eval.retrieval import build_retriever
from rag_eval.retrieval.vector_store import VectorStore


def test_evaluate_end_to_end_with_fake_embeddings(tmp_path, sample_chunks, fake_embedder):
    cfg = Config(name="unit", pdf_paths=[])
    cfg.bm25.tokenizer = "whitespace"
    r = build_retriever(cfg, sample_chunks, embedder=fake_embedder, store=VectorStore(name="eval_unit"), index=True)
    items = [
        EvalItem("q001", "재생에너지 전환율은?", "24.8%", [25], "number", "dev"),
        EvalItem("q002", "온실가스 배출량 2024년", "14,889천 톤", [68], "number", "dev"),
        EvalItem("q003", "존재하지 않는 내용", "-", [99], "text", "dev"),
    ]
    results = evaluate(r, items, top_k=3, weight_vector=0.5)
    assert [x.first_hit_rank for x in results[:2]] == [1, 1]
    assert results[2].first_hit_rank is None
    assert len(results[0].top_pages) == 3 and len(results[0].top_chunk_ids) == 3

    summaries = summarize(results, (1, 3))
    paths = write_results(tmp_path / "out", results, summaries, (1, 3), {"w": 0.5})
    with open(paths["per_query"], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["id"] for row in rows] == ["q001", "q002", "q003"]
    assert rows[0]["hit@1"] == "1" and rows[2]["hit@3"] == "0" and rows[2]["first_hit_rank"] == ""
    assert "| group | n | recall@1 | recall@3 | mrr |" in paths["summary_md"].read_text(encoding="utf-8")
    assert [m.id for m in missed_questions(results, 3)] == ["q003"]
    assert markdown_table([]) == "(no rows)"


def test_run_command_defaults_to_dev_split():
    from rag_eval.cli import build_parser
    args = build_parser().parse_args(["run", "--config", "x.yaml"])
    assert args.split == "dev"
    args = build_parser().parse_args(["run", "--config", "x.yaml", "--split", "heldout"])
    assert args.split == "heldout"

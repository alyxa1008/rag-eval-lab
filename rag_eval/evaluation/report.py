"""Write per-query and summary results (CSV + Markdown) and format them for the terminal."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rag_eval.evaluation.metrics import GroupSummary, QueryResult


def per_query_rows(results: list[QueryResult], ks: tuple[int, ...]) -> list[dict[str, object]]:
    rows = []
    for r in results:
        row: dict[str, object] = {
            "id": r.id, "type": r.type, "split": r.split,
            "gold_pages": ";".join(map(str, r.gold_pages)),
            "first_hit_rank": r.first_hit_rank if r.first_hit_rank is not None else "",
        }
        for k in ks:
            row[f"hit@{k}"] = int(r.hit_at(k))
        row["top_pages"] = ";".join(map(str, r.top_pages))
        row["top_chunk_ids"] = ";".join(r.top_chunk_ids)
        row["question"] = r.question
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def write_results(out_dir: Path, results: list[QueryResult], summaries: list[GroupSummary],
                  ks: tuple[int, ...], run_info: dict[str, object]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [s.as_row(ks) for s in summaries]
    paths = {
        "per_query": out_dir / "per_query.csv",
        "summary": out_dir / "summary.csv",
        "summary_md": out_dir / "summary.md",
        "run_info": out_dir / "run_info.json",
    }
    _write_csv(paths["per_query"], per_query_rows(results, ks))
    _write_csv(paths["summary"], summary_rows)
    paths["summary_md"].write_text(markdown_table(summary_rows) + "\n", encoding="utf-8")
    paths["run_info"].write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def missed_questions(results: list[QueryResult], k: int) -> list[QueryResult]:
    return [r for r in results if not r.hit_at(k)]

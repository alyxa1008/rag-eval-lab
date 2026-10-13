"""`answer-hit`: recompute AnsHit@k (chunk contains the answer) for every run of a split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rag_eval.chunking import load_chunks
from rag_eval.evalset import load_eval_set
from rag_eval.evaluation.answer_hit import answer_hit_rank
from rag_eval.evaluation.report import markdown_table


def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    items = {it.id: it for it in load_eval_set(args.eval_set)}
    ks = (1, 3, 5)
    rows = []
    chunk_cache: dict[str, dict] = {}
    for per_query in sorted(results_dir.glob(f"*/{args.split}/*/per_query.csv")):
        run_dir = per_query.parent
        cfg_name = run_dir.parent.parent.name
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        if cfg_name not in chunk_cache:
            chunk_cache[cfg_name] = {c.id: c for c in load_chunks(Path(args.processed_dir) / cfg_name / "chunks.jsonl")}
        chunks = chunk_cache[cfg_name]
        with open(per_query, encoding="utf-8", newline="") as f:
            queries = list(csv.DictReader(f))
        ranks, page_ranks = [], []
        for q in queries:
            ids = q["top_chunk_ids"].split(";") if q["top_chunk_ids"] else []
            texts = [chunks[i].text for i in ids]
            pages = [chunks[i].page for i in ids]
            ranks.append(answer_hit_rank(pages, texts, items[q["id"]].gold_pages, items[q["id"]].answer))
            page_ranks.append(int(q["first_hit_rank"]) if q["first_hit_rank"] else None)
        n = len(queries)
        row: dict[str, object] = {"config": cfg_name, "run": run_dir.name,
                                  "chunk": (info.get("chunking") or {}).get("max_chars", ""), "n": n}
        for k in ks:
            row[f"Page@{k}"] = round(sum(1 for r in page_ranks if r and r <= k) / n, 3)
            row[f"Ans@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
        row["AnsMRR"] = round(sum(1 / r for r in ranks if r) / n, 3)
        rows.append(row)
        with open(run_dir / "answer_hit.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "page_rank", "answer_rank"])
            w.writerows([(q["id"], pr or "", ar or "") for q, pr, ar in zip(queries, page_ranks, ranks)])
    if not rows:
        print("no runs found")
        return 1
    rows.sort(key=lambda r: (-r["AnsMRR"], -r["Ans@1"], str(r["config"]), str(r["run"])))
    table = markdown_table(rows)
    out = results_dir / f"answer_hit_{args.split}.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {out}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("answer-hit", help="chunk-level AnsHit@k for all runs of a split (no retrieval re-run)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--eval-set", default="data/eval_set.csv")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.set_defaults(func=run)

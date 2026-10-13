"""`compare`: gather every run's summary under results/ into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rag_eval.evaluation.report import markdown_table


def collect_runs(results_dir: str | Path, split: str = "dev") -> list[dict[str, object]]:
    rows = []
    for summary in sorted(Path(results_dir).glob(f"*/{split}/*/summary.csv")):
        run_dir = summary.parent
        info_path = run_dir / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        with open(summary, encoding="utf-8", newline="") as f:
            groups = {r["group"]: r for r in csv.DictReader(f)}
        overall = groups.get("all")
        if not overall:
            continue
        row: dict[str, object] = {
            "config": run_dir.parent.parent.name,
            "run": run_dir.name,
            "chunk": (info.get("chunking") or {}).get("max_chars", ""),
            "w": info.get("weight_vector", ""),
            "rerank": "on" if info.get("rerank") else "off",
            "n": int(overall["n"]),
        }
        for key in overall:
            if key.startswith("recall@") or key == "mrr":
                row[key] = float(overall[key])
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> int:
    rows = collect_runs(args.results_dir, args.split)
    if not rows:
        print(f"no runs found under {args.results_dir}/*/{args.split}/")
        return 1
    rows.sort(key=lambda r: (-r.get("mrr", 0.0), -r.get("recall@1", 0.0), str(r["config"]), str(r["run"])))
    table = markdown_table(rows)
    out = Path(args.results_dir) / f"compare_{args.split}.md"
    out.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {out}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("compare", help="one table across all runs of a split (sorted by MRR)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.set_defaults(func=run)

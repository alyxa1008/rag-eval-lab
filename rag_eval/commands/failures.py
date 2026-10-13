"""`failures`: for one run, show questions missed at rank k with where the gold pages actually landed."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_eval.chunking import load_chunks
from rag_eval.config import load_config


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    per_query = Path(cfg.eval.results_dir) / cfg.name / args.split / args.run / "per_query.csv"
    if not per_query.exists():
        print(f"{per_query} not found")
        return 1
    chunks = {c.id: c for c in load_chunks(cfg.output_dir / "chunks.jsonl")}
    with open(per_query, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missed = [r for r in rows if not r["first_hit_rank"] or int(r["first_hit_rank"]) > args.k]
    print(f"{cfg.name}/{args.split}/{args.run}: {len(missed)} of {len(rows)} missed @{args.k}\n")
    for r in missed:
        gold = r["gold_pages"].replace(";", ",")
        rank = r["first_hit_rank"] or "none"
        print(f"{r['id']} [{r['type']}] gold p{gold} | first hit rank={rank}")
        print(f"   Q: {r['question']}")
        for i, cid in enumerate(r["top_chunk_ids"].split(";")[:args.show], start=1):
            c = chunks.get(cid)
            if c is None:
                continue
            mark = "*" if str(c.page) in r["gold_pages"].split(";") else " "
            print(f"   {mark}{i:>2}. p{c.page:<3} {c.type:<5} {c.text.replace(chr(10), ' ')[:args.width]}")
        print()
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("failures", help="list missed questions of a run with top chunks (failure analysis)")
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True, help="run folder name, e.g. w0.6 or w0.6_rerank")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("-k", type=int, default=1, help="miss threshold rank (default 1)")
    p.add_argument("--show", type=int, default=3, help="top chunks to print per question")
    p.add_argument("--width", type=int, default=80)
    p.set_defaults(func=run)

"""`run`: evaluate one config on the eval set. Defaults to the dev split; heldout must be explicit."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

from rag_eval.chunking import load_chunks
from rag_eval.config import load_config
from rag_eval.evalset import load_eval_set
from rag_eval.evaluation.metrics import summarize
from rag_eval.evaluation.report import markdown_table, missed_questions, write_results
from rag_eval.evaluation.runner import evaluate
from rag_eval.retrieval import build_retriever


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    split = args.split
    if split == "heldout":
        print("!! HELDOUT split: use only once, after all tuning on dev is finished.\n")

    items = load_eval_set(args.eval_set or cfg.eval.eval_set, split=split)
    if not items:
        print(f"no items with split={split} in the eval set; fill data/eval_set.csv first")
        return 1

    chunks_path = cfg.output_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"{chunks_path} not found; run `index` first")
        return 1
    chunks = load_chunks(chunks_path)
    w = cfg.retrieval.weight_vector if args.w is None else args.w
    retriever = build_retriever(cfg, chunks, with_vector=w > 0)

    results = evaluate(retriever, items, top_k=cfg.eval.top_k, weight_vector=w)
    summaries = summarize(results, cfg.eval.ks)

    run_name = args.name or f"w{w:g}"
    out_dir = Path(cfg.eval.results_dir) / cfg.name / split / run_name
    run_info = {
        "config": str(args.config), "config_name": cfg.name, "split": split, "weight_vector": w,
        "embedding": f"{cfg.embedding.provider}:{cfg.embedding.model}", "tokenizer": cfg.bm25.tokenizer,
        "chunking": vars(cfg.chunking), "top_k": cfg.eval.top_k, "n_items": len(items),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    paths = write_results(out_dir, results, summaries, cfg.eval.ks, run_info)

    print(f"config={cfg.name} split={split} w={w:g} n={len(items)} "
          f"embedding={run_info['embedding']} tokenizer={cfg.bm25.tokenizer}\n")
    print(markdown_table([s.as_row(cfg.eval.ks) for s in summaries]))
    k_show = max(cfg.eval.ks)
    missed = missed_questions(results, k_show)
    if missed:
        print(f"\nmissed @{k_show} ({len(missed)}):")
        for r in missed:
            print(f"  {r.id} [{r.type}] gold={r.gold_pages} top={r.top_pages[:k_show]}  {r.question}")
    print(f"\nwrote {paths['per_query']} and {paths['summary']}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="evaluate a config: Recall@k / MRR on the dev split (M2)")
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["dev", "heldout"], default="dev",
                   help="dev (default). heldout only for the single final run")
    p.add_argument("--eval-set", default=None, help="override eval set path")
    p.add_argument("-w", type=float, default=None, help="override vector weight for this run")
    p.add_argument("--name", default=None, help="run name for the results folder (default: w<weight>)")
    p.set_defaults(func=run)

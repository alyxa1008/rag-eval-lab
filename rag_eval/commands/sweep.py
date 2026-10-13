"""`sweep`: evaluate several vector weights on the DEV split with one shared retriever."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag_eval.chunking import load_chunks
from rag_eval.config import load_config
from rag_eval.evalset import load_eval_set
from rag_eval.evaluation.metrics import summarize
from rag_eval.evaluation.report import markdown_table, write_results
from rag_eval.evaluation.runner import evaluate
from rag_eval.retrieval import build_retriever

DEFAULT_WEIGHTS = "0,0.3,0.5,0.6,0.7,1.0"


def sweep_row(run: str, w: float, rerank: bool, summaries, ks) -> dict[str, object]:
    overall = summaries[0]
    row: dict[str, object] = {"run": run, "w": w, "rerank": "on" if rerank else "off", "n": overall.n}
    for k in ks:
        row[f"R@{k}"] = round(overall.recall[k], 3)
    row["MRR"] = round(overall.mrr, 3)
    for g in summaries[1:]:
        row[f"{g.group.removeprefix('type=')} R@1"] = round(g.recall[1], 3)
    return row


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    weights = [float(x) for x in args.weights.split(",")]
    rerank = cfg.rerank.enabled if args.rerank is None else args.rerank

    items = load_eval_set(cfg.eval.eval_set, split="dev")   # sweeps are dev-only by design
    chunks_path = cfg.output_dir / "chunks.jsonl"
    if not items or not chunks_path.exists():
        print("need a filled eval set and an indexed config (run `index` first)")
        return 1
    chunks = load_chunks(chunks_path)
    retriever = build_retriever(cfg, chunks, with_vector=any(w > 0 for w in weights), with_reranker=rerank)

    rows = []
    base_dir = Path(cfg.eval.results_dir) / cfg.name / "dev"
    for w in weights:
        results = evaluate(retriever, items, top_k=cfg.eval.top_k, weight_vector=w, rerank=rerank)
        summaries = summarize(results, cfg.eval.ks)
        run_name = f"w{w:g}" + ("_rerank" if rerank else "")
        write_results(base_dir / run_name, results, summaries, cfg.eval.ks, {
            "config": str(args.config), "config_name": cfg.name, "split": "dev", "weight_vector": w,
            "rerank": cfg.rerank.model if rerank else None, "rerank_top_n": cfg.rerank.top_n,
            "embedding": f"{cfg.embedding.provider}:{cfg.embedding.model}", "tokenizer": cfg.bm25.tokenizer,
            "chunking": vars(cfg.chunking), "top_k": cfg.eval.top_k, "n_items": len(items),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        rows.append(sweep_row(run_name, w, rerank, summaries, cfg.eval.ks))
        print(f"  w={w:<4g} rerank={'on ' if rerank else 'off'} R@1={rows[-1]['R@1']:.3f} MRR={rows[-1]['MRR']:.3f}")

    table = markdown_table(rows)
    out = base_dir / ("sweep_rerank.md" if rerank else "sweep.md")
    out.write_text(f"config={cfg.name} split=dev n={len(items)} embedding={cfg.embedding.model} "
                   f"chunk={cfg.chunking.max_chars}\n\n{table}\n", encoding="utf-8")
    print(f"\n{table}\n\nwrote {out}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sweep", help="evaluate several vector weights on the dev split (M3)")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"comma-separated, default {DEFAULT_WEIGHTS}")
    p.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.set_defaults(func=run)

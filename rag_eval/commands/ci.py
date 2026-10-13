"""`ci`: bootstrap confidence intervals for runs and paired differences against a reference run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_eval.evaluation.bootstrap import bootstrap_ci, paired_diff_ci
from rag_eval.evaluation.report import markdown_table

DEFAULT_RUNS = "chunk1500:w0.6,base:w0.6,base:w0.7,base:w0.5,base:w0.6_rerank,base:w1,base:w0"


def _ranks(results_dir: Path, split: str, spec: str) -> tuple[list[str], list[int]]:
    cfg, run = spec.split(":")
    with open(results_dir / cfg / split / run / "per_query.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["id"] for r in rows], [int(r["first_hit_rank"]) if r["first_hit_rank"] else 0 for r in rows]


def run(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    specs = args.runs.split(",")
    ref_ids, ref = _ranks(results_dir, args.split, specs[0])
    rows = []
    for spec in specs:
        ids, ranks = _ranks(results_dir, args.split, spec)
        if ids != ref_ids:
            print(f"{spec}: question set differs from {specs[0]}; skipping")
            continue
        ci = bootstrap_ci(ranks, k=1, n_boot=args.n_boot, seed=args.seed)
        row: dict[str, object] = {
            "run": spec, "n": len(ranks),
            "R@1": f"{ci['recall'][0]:.3f} [{ci['recall'][1]:.2f}, {ci['recall'][2]:.2f}]",
            "MRR": f"{ci['mrr'][0]:.3f} [{ci['mrr'][1]:.2f}, {ci['mrr'][2]:.2f}]",
        }
        if spec == specs[0]:
            row["ΔR@1 vs ref"] = "(reference)"
            row["ΔMRR vs ref"] = ""
            row["P(ref better)"] = ""
        else:
            d = paired_diff_ci(ranks, ref, k=1, n_boot=args.n_boot, seed=args.seed)   # ref - this
            row["ΔR@1 vs ref"] = f"{-d['recall'][0]:+.3f} [{-d['recall'][2]:+.2f}, {-d['recall'][1]:+.2f}]"
            row["ΔMRR vs ref"] = f"{-d['mrr'][0]:+.3f} [{-d['mrr'][2]:+.2f}, {-d['mrr'][1]:+.2f}]"
            row["P(ref better)"] = f"{d['mrr'][3]:.2f}"
        rows.append(row)
    table = markdown_table(rows)
    out = results_dir / (f"ci_{args.split}_{args.name}.md" if args.name else f"ci_{args.split}.md")
    out.write_text(f"bootstrap n_boot={args.n_boot} seed={args.seed}; Δ = run − reference (paired over questions); "
                   f"P = share of resamples where the reference has higher MRR\n\n{table}\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {out}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("ci", help="bootstrap 95% CIs and paired differences vs the first run")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--runs", default=DEFAULT_RUNS, help="comma-separated config:run, first = reference")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="suffix for the output file name")
    p.set_defaults(func=run)

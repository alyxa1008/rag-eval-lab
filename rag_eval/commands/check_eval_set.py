"""`check-eval-set`: validate data/eval_set.csv format and print split/type counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.evalset import load_eval_set, read_rows, summarize, validate_rows


def run(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        rows = read_rows(path)
    except (OSError, ValueError) as e:
        print(f"error: {e}")
        return 1
    errors = validate_rows(rows)
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print("  -", e)
        return 1
    items = load_eval_set(path)
    counts = summarize(items)
    total = len(items)
    heldout = counts["split"].get("heldout", 0)
    print(f"ok: {total} items")
    print(f"  split: {dict(counts['split'])}" + (f"  (heldout {heldout / total:.0%})" if total else ""))
    print(f"  type:  {dict(counts['type'])}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("check-eval-set", help="validate data/eval_set.csv format")
    p.add_argument("--path", default="data/eval_set.csv")
    p.set_defaults(func=run)

"""Command-line entry point: `python -m rag_eval <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_eval.chunking import chunk_pages
from rag_eval.config import load_config
from rag_eval.evalset import load_eval_set, read_rows, summarize, validate_rows
from rag_eval.parsing import parse_pdf
from rag_eval.summary import HEADERS, format_table, page_summary_rows


def cmd_parse(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    all_chunks = []
    for pdf in cfg.pdf_paths:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"[skip] {pdf_path} not found (put the PDF under data/raw/)")
            continue
        pages = parse_pdf(pdf_path, cfg.parsing)
        chunks = chunk_pages(pages, cfg.chunking, source=pdf_path.stem)
        all_chunks.extend(chunks)

        print(f"\n== {pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks ==")
        if args.page:
            for c in chunks:
                if c.page == args.page:
                    print(f"\n--- {c.id} [{c.type}] {c.meta}\n{c.text}")
        else:
            rows = page_summary_rows(pages, chunks)
            print(format_table(HEADERS, rows))
            print(f"\ntotal: tables(lines)={sum(r[2] for r in rows)} tables(layout)={sum(r[3] for r in rows)} "
                  f"table_rows={sum(r[4] for r in rows)} "
                  f"text_chunks={sum(r[5] for r in rows)} table_chunks={sum(r[6] for r in rows)}")

    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"\nwrote {len(all_chunks)} chunks -> {out_path}")
    return 0


def cmd_check_eval_set(args: argparse.Namespace) -> int:
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
    print(f"  split: {dict(counts['split'])}"
          + (f"  (heldout {heldout / total:.0%})" if total else ""))
    print(f"  type:  {dict(counts['type'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_eval")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("parse", help="parse PDFs into chunks and print a per-page report (M0)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--page", type=int, help="dump all chunks of this page instead of the summary")
    sp.set_defaults(func=cmd_parse)

    sc = sub.add_parser("check-eval-set", help="validate data/eval_set.csv format")
    sc.add_argument("--path", default="data/eval_set.csv")
    sc.set_defaults(func=cmd_check_eval_set)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

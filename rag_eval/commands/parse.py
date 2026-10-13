"""`parse`: PDFs -> chunks.jsonl plus a per-page report (M0 check)."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_eval.chunking import Chunk, chunk_pages, save_chunks
from rag_eval.config import Config, load_config
from rag_eval.parsing import parse_pdf
from rag_eval.summary import HEADERS, format_table, page_summary_rows


def build_chunks(cfg: Config, verbose: bool = True, dump_page: int | None = None) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for pdf in cfg.pdf_paths:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"[skip] {pdf_path} not found (put the PDF under data/raw/)")
            continue
        pages = parse_pdf(pdf_path, cfg.parsing)
        chunks = chunk_pages(pages, cfg.chunking, source=pdf_path.stem)
        all_chunks.extend(chunks)
        if not verbose:
            continue
        print(f"\n== {pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks ==")
        if dump_page:
            for c in chunks:
                if c.page == dump_page:
                    print(f"\n--- {c.id} [{c.type}] {c.meta}\n{c.text}")
        else:
            rows = page_summary_rows(pages, chunks)
            print(format_table(HEADERS, rows))
            print(f"\ntotal: tables(lines)={sum(r[2] for r in rows)} tables(layout)={sum(r[3] for r in rows)} "
                  f"table_rows={sum(r[4] for r in rows)} "
                  f"text_chunks={sum(r[5] for r in rows)} table_chunks={sum(r[6] for r in rows)}")
    return all_chunks


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    chunks = build_chunks(cfg, dump_page=args.page)
    out_path = cfg.output_dir / "chunks.jsonl"
    save_chunks(chunks, out_path)
    print(f"\nwrote {len(chunks)} chunks -> {out_path}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("parse", help="parse PDFs into chunks and print a per-page report (M0)")
    p.add_argument("--config", required=True)
    p.add_argument("--page", type=int, help="dump all chunks of this page instead of the summary")
    p.set_defaults(func=run)

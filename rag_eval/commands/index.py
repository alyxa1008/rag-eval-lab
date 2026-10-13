"""`index`: parse + chunk + embed (cached) + upsert into ChromaDB."""

from __future__ import annotations

import argparse

from rag_eval.chunking import save_chunks
from rag_eval.commands.parse import build_chunks
from rag_eval.config import load_config
from rag_eval.retrieval import build_retriever


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    chunks = build_chunks(cfg, verbose=False)
    if not chunks:
        print("no chunks produced; check pdf_paths in the config")
        return 1
    save_chunks(chunks, cfg.output_dir / "chunks.jsonl")
    retriever = build_retriever(cfg, chunks, index=True)
    emb = retriever.embedder
    print(f"indexed {len(chunks)} chunks -> collection for '{cfg.name}' ({retriever.store.count()} vectors)")
    print(f"embedding '{emb.model}': cache {emb.hits} hits / {emb.misses} misses, {emb.inner.calls} model call(s)")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("index", help="build the vector + BM25 index for a config (M1)")
    p.add_argument("--config", required=True)
    p.set_defaults(func=run)

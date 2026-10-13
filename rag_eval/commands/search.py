"""`search`: ad-hoc query against an indexed config, printing page and scores per hit."""

from __future__ import annotations

import argparse

from rag_eval.chunking import load_chunks
from rag_eval.config import load_config
from rag_eval.retrieval import build_retriever


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    chunks_path = cfg.output_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"{chunks_path} not found; run `parse` or `index` first")
        return 1
    chunks = load_chunks(chunks_path)
    w = {"bm25": 0.0, "vector": 1.0, "hybrid": cfg.retrieval.weight_vector}[args.mode]
    if args.w is not None:
        w = args.w
    retriever = build_retriever(cfg, chunks, with_vector=w > 0)
    hits = retriever.search(args.query, k=args.k, weight_vector=w)
    print(f"query: {args.query}   (mode={args.mode}, w={w}, k={args.k})")
    for h in hits:
        vs = f"{h.vector_score:.3f}" if h.vector_score is not None else "  -  "
        bs = f"{h.bm25_score:.2f}" if h.bm25_score is not None else "  -  "
        preview = h.chunk.text.replace("\n", " ")[:args.width]
        print(f"{h.rank:>2}. p{h.chunk.page:<3} {h.chunk.type:<5} score={h.score:.3f} vec={vs} bm25={bs} | {preview}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("search", help="run one query and print ranked chunks with pages")
    p.add_argument("--config", required=True)
    p.add_argument("-q", "--query", required=True)
    p.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    p.add_argument("-w", type=float, default=None, help="override vector weight (0=BM25 only, 1=vector only)")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--width", type=int, default=90, help="preview characters per hit")
    p.set_defaults(func=run)

"""YAML config loading. All tunables live in configs/*.yaml, never in code."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ChunkingConfig:
    max_chars: int = 800
    overlap_chars: int = 100
    table_rows_per_chunk: int = 10


@dataclass
class ParsingConfig:
    layout_tables: bool = True   # reconstruct borderless tables from word coordinates
    row_tol: float = 4.0
    cell_gap: float = 8.0
    min_values: int = 2


@dataclass
class EmbeddingConfig:
    provider: str = "local"      # "local" (sentence-transformers) | "openai"
    model: str = "BAAI/bge-m3"
    cache_dir: str = ".cache/embeddings"
    batch_size: int = 16


@dataclass
class BM25Config:
    tokenizer: str = "kiwi"      # "kiwi" (morphemes) | "whitespace"
    k1: float = 1.5
    b: float = 0.75


@dataclass
class RetrievalConfig:
    weight_vector: float = 0.6   # w in  w*vector + (1-w)*bm25
    top_k: int = 10
    candidate_k: int = 50        # per-retriever pool size before fusion
    vector_store_dir: str = "chroma_db"


@dataclass
class Config:
    name: str
    pdf_paths: list[str]
    seed: int = 42
    processed_dir: str = "data/processed"
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def output_dir(self) -> Path:
        return Path(self.processed_dir) / self.name


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    missing = [k for k in ("name", "pdf_paths") if k not in raw]
    if missing:
        raise ValueError(f"config {path}: missing required keys {missing}")
    return Config(
        name=str(raw["name"]),
        pdf_paths=[str(p) for p in raw["pdf_paths"]],
        seed=int(raw.get("seed", 42)),
        processed_dir=str(raw.get("processed_dir", "data/processed")),
        parsing=ParsingConfig(**(raw.get("parsing") or {})),
        chunking=ChunkingConfig(**(raw.get("chunking") or {})),
        embedding=EmbeddingConfig(**(raw.get("embedding") or {})),
        bm25=BM25Config(**(raw.get("bm25") or {})),
        retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
    )

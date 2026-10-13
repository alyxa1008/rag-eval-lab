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
class Config:
    name: str
    pdf_paths: list[str]
    seed: int = 42
    processed_dir: str = "data/processed"
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)

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
    )

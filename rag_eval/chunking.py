"""Page-based text chunks + separate table chunks. Every chunk carries `page`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_eval.config import ChunkingConfig
from rag_eval.parsing.models import Page, Table
from rag_eval.parsing.tables import rows_to_text

_BREAKS = ("\n\n", "\n", ". ", " ")


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    type: str                      # "text" | "table"
    source: str                    # document name (pdf stem)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_text(text: str, max_chars: int, overlap_chars: int = 0) -> list[str]:
    """Split text into windows of at most `max_chars`, preferring natural breaks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            floor = int(max_chars * 0.5)
            for sep in _BREAKS:
                cut = window.rfind(sep)
                if cut >= floor:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def _table_chunks(table: Table, source: str, rows_per_chunk: int) -> list[Chunk]:
    if rows_per_chunk <= 0:
        raise ValueError("table_rows_per_chunk must be positive")
    parts = [table.rows[i:i + rows_per_chunk] for i in range(0, table.n_rows, rows_per_chunk)]
    chunks: list[Chunk] = []
    for part_no, rows in enumerate(parts, start=1):
        row_start = (part_no - 1) * rows_per_chunk
        title = f" {table.title}" if table.title else ""
        label = f"[표 p{table.page}-{table.index + 1}{title} ({part_no}/{len(parts)})]"
        text = f"{label}\n" + rows_to_text(table.header, rows)
        chunks.append(Chunk(
            id=f"{source}-p{table.page:03d}-tb{table.index + 1}-{part_no}",
            text=text,
            page=table.page,
            type="table",
            source=source,
            meta={
                "table_index": table.index,
                "table_kind": table.kind,
                "table_title": table.title,
                "part": part_no,
                "n_parts": len(parts),
                "row_start": row_start,
                "row_end": row_start + len(rows),
                "n_rows_total": table.n_rows,
            },
        ))
    return chunks


def chunk_pages(pages: list[Page], cfg: ChunkingConfig, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = split_text(page.text, cfg.max_chars, cfg.overlap_chars)
        for k, piece in enumerate(pieces, start=1):
            chunks.append(Chunk(
                id=f"{source}-p{page.page:03d}-t{k}",
                text=piece,
                page=page.page,
                type="text",
                source=source,
                meta={"part": k, "n_parts": len(pieces)},
            ))
        for table in page.tables:
            chunks.extend(_table_chunks(table, source, cfg.table_rows_per_chunk))
    return chunks


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]

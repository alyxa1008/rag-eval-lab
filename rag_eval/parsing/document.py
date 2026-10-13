"""Combine body text and tables into per-page records."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from rag_eval.config import ParsingConfig
from rag_eval.parsing.layout_tables import LayoutConfig, extract_layout_tables_from_words, page_words
from rag_eval.parsing.models import Page
from rag_eval.parsing.tables import extract_tables
from rag_eval.parsing.text import extract_page_texts


def parse_pdf(pdf_path: str | Path, cfg: ParsingConfig | None = None) -> list[Page]:
    """Parse a PDF into pages.

    Body text comes from PyMuPDF. Ruled tables come from pdfplumber; borderless
    numeric tables are reconstructed from word coordinates when enabled.
    Body text still contains the table cells as flat text; table chunks are added
    on top of it rather than cut out of it.
    """
    cfg = cfg or ParsingConfig()
    texts = extract_page_texts(pdf_path)
    tables = extract_tables(pdf_path)
    if cfg.layout_tables:
        layout_cfg = LayoutConfig(row_tol=cfg.row_tol, cell_gap=cfg.cell_gap, min_values=cfg.min_values)
        with pymupdf.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc, start=1):
                existing = tables.setdefault(i, [])
                existing.extend(extract_layout_tables_from_words(
                    page_words(page), page=i, cfg=layout_cfg, index_start=len(existing)))
    return [
        Page(page=i, text=text, tables=tables.get(i, []))
        for i, text in enumerate(texts, start=1)
    ]

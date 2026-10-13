"""Table extraction with pdfplumber and row-wise textualization."""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from rag_eval.parsing.models import Table

RawTable = list[list[str | None]]


def _clean_cell(cell: str | None) -> str:
    return " ".join((cell or "").split())


def clean_table(raw: RawTable, page: int, index: int, min_rows: int = 2, min_cols: int = 2) -> Table | None:
    """Turn a raw pdfplumber table into a Table; drop empty rows and noise tables."""
    rows = [[_clean_cell(c) for c in row] for row in raw if row]
    rows = [r for r in rows if any(r)]
    if len(rows) < min_rows:
        return None
    width = max(len(r) for r in rows)
    if width < min_cols:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    header = [h if h else f"col{i + 1}" for i, h in enumerate(header)]
    return Table(page=page, index=index, header=header, rows=body)


def extract_tables(pdf_path: str | Path) -> dict[int, list[Table]]:
    """Map 1-based page number -> tables found on that page."""
    found: dict[int, list[Table]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tables: list[Table] = []
            for raw in page.extract_tables():
                table = clean_table(raw, page=page_no, index=len(tables))
                if table is not None:
                    tables.append(table)
            if tables:
                found[page_no] = tables
    return found


def row_to_text(header: list[str], row: list[str]) -> str:
    """One row -> '헤더1: 값1 | 헤더2: 값2' (empty cells skipped)."""
    pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
    return " | ".join(pairs)


def rows_to_text(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(row_to_text(header, r) for r in rows)

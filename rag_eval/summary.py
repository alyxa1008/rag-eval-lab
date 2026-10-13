"""Per-page parsing report used by the M0 check."""

from __future__ import annotations

from rag_eval.chunking import Chunk
from rag_eval.parsing.models import Page

HEADERS = ["page", "chars", "tbl_lines", "tbl_layout", "table_rows", "text_chunks", "table_chunks"]


def page_summary_rows(pages: list[Page], chunks: list[Chunk]) -> list[list[int]]:
    by_page: dict[int, dict[str, int]] = {p.page: {"text": 0, "table": 0} for p in pages}
    for c in chunks:
        by_page.setdefault(c.page, {"text": 0, "table": 0})[c.type] += 1
    rows = []
    for p in pages:
        counts = by_page[p.page]
        rows.append([
            p.page,
            len(p.text),
            sum(1 for t in p.tables if t.kind == "lines"),
            sum(1 for t in p.tables if t.kind == "layout"),
            sum(t.n_rows for t in p.tables),
            counts["text"],
            counts["table"],
        ])
    return rows


def format_table(headers: list[str], rows: list[list[object]]) -> str:
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)]
    line = lambda r: " | ".join(c.rjust(w) for c, w in zip(r, widths))  # noqa: E731
    out = [line(headers), "-+-".join("-" * w for w in widths)]
    out.extend(line(r) for r in cells)
    return "\n".join(out)

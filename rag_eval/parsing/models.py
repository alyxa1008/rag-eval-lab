from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Table:
    page: int                 # 1-based page number
    index: int                # order of the table within the page (0-based)
    header: list[str]
    rows: list[list[str]]
    title: str = ""           # table caption if known (layout tables)
    kind: str = "lines"       # "lines" (ruled, pdfplumber) | "layout" (word coordinates)

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class Page:
    page: int                 # 1-based page number
    text: str
    tables: list[Table] = field(default_factory=list)

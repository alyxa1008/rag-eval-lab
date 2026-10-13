"""Reconstruct borderless tables from word coordinates.

ESG "Facts & Figures" pages lay out numbers in columns without ruling lines, so
line-based table detection finds nothing. Here we cluster words into visual rows,
cut each row into cells by horizontal gaps, and treat a row whose trailing cells
are numbers as a data row. A row holding several year tokens (2022년 2023년 …)
is a header that names the value columns for the data rows beneath it.

Side-by-side tables are handled with "bands": each header opens a column band,
and later cells are routed to the band they overlap (or the nearest one). Within
a band, a run of values followed by text, or a repeated year, starts a new segment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_eval.parsing.models import Table

_NUM = r"[-–−]?\d[\d,]*(?:\.\d+)?%?"
VALUE_RE = re.compile(rf"^(?:{_NUM}|\({_NUM}\)|[-–−])$")
YEAR_RE = re.compile(r"^(?:FY)?(?:19|20)\d{2}년?$")
FOOTNOTE_RE = re.compile(r"^\d\)(?:,\s*\d\))*,?$")                    # standalone: "1)", "1),2)", "3),"
FOOTNOTE_SUFFIX_RE = re.compile(r"^(.*?[^\d\s])(\d\)(?:,\d\))*,?)$")  # glued: "임직원1)"

Word = tuple[float, float, float, float, str]   # x0, y0, x1, y1, text
Cell = tuple[str, float, float]                 # text, x0, x1


@dataclass
class LayoutConfig:
    row_tol: float = 4.0        # max y0 difference for words on the same row
    cell_gap: float = 8.0       # horizontal gap that separates two cells
    min_values: int = 2         # trailing numeric cells needed for a data row
    max_unit_chars: int = 12
    max_section_chars: int = 40


@dataclass
class _Segment:
    cells: list[str]
    x0: float
    x1: float
    n_values: int = 0           # trailing value cells

    @property
    def years(self) -> list[str]:
        return [c for c in self.cells if YEAR_RE.match(c)]

    @property
    def is_header(self) -> bool:
        return len(self.years) >= 2

    @property
    def title(self) -> str:
        return clean_label([c for c in self.cells if not YEAR_RE.match(c)])


@dataclass
class _Band:
    """Running state for one column band (one stack of tables) on the page."""
    header: list[str]
    title: str
    x0: float
    x1: float
    section: str = ""
    rows: list[list[str]] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)

    def grow(self, x0: float, x1: float) -> None:
        self.x0, self.x1 = min(self.x0, x0), max(self.x1, x1)


# --- text helpers -----------------------------------------------------------

def _balanced(text: str) -> bool:
    return text.count("(") == text.count(")")


def clean_label(cells: list[str]) -> str:
    """Join cells and drop footnote markers ("1)", "1),2)", "임직원1)").

    A marker is only dropped when the label stays paren-balanced without it, so
    "(Scope 1)" keeps its "1)" while "총 임직원1)" loses it.
    """
    kept: list[str] = []
    for w in " ".join(cells).split():
        if FOOTNOTE_RE.match(w) and _balanced(" ".join(kept)):
            continue
        m = FOOTNOTE_SUFFIX_RE.match(w)
        if m and _balanced(" ".join(kept + [m.group(1)])):
            w = m.group(1)
        kept.append(w)
    return " ".join(kept).strip()


def _is_section(label: str, cfg: LayoutConfig) -> bool:
    return (0 < len(label) <= cfg.max_section_chars
            and not label[:1].isdigit()
            and label[-1] not in ").,:")


# --- geometry: rows, cells, segments ---------------------------------------

def cluster_rows(words: list[Word], row_tol: float) -> list[list[Word]]:
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if rows and abs(w[1] - rows[-1][-1][1]) <= row_tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w[0]) for r in rows]


def split_cells(row: list[Word], cell_gap: float) -> list[Cell]:
    """Merge adjacent words into cells; returns (text, x0, x1) per cell."""
    cells: list[Cell] = []
    for x0, _, x1, _, text in row:
        if cells and x0 - cells[-1][2] <= cell_gap:
            t, cx0, _ = cells[-1]
            cells[-1] = (f"{t} {text}", cx0, x1)
        else:
            cells.append((text, x0, x1))
    return cells


def _make_segment(cells: list[Cell]) -> _Segment:
    texts = [c[0] for c in cells]
    n_values = 0
    for t in reversed(texts):
        if not VALUE_RE.match(t):
            break
        n_values += 1
    return _Segment(cells=texts, x0=cells[0][1], x1=cells[-1][2], n_values=n_values)


def split_segments(cells: list[Cell], min_values: int) -> list[_Segment]:
    """Cut a row into table rows: a run of values followed by text, or a repeated year."""
    segments: list[_Segment] = []
    current: list[Cell] = []
    run = 0
    years_seen: set[str] = set()
    for cell in cells:
        text = cell[0]
        is_year = bool(YEAR_RE.match(text))
        is_value = is_year or bool(VALUE_RE.match(text))
        new_table = (not is_value and run >= min_values) or (is_year and text in years_seen)
        if new_table and current:
            segments.append(_make_segment(current))
            current, years_seen = [], set()
        current.append(cell)
        run = run + 1 if is_value else 0
        if is_year:
            years_seen.add(text)
    if current:
        segments.append(_make_segment(current))
    return segments


# --- bands ------------------------------------------------------------------

def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return min(a1, b1) - max(a0, b0) > 0


def _band_for(x0: float, x1: float, bands: list[_Band]) -> tuple[_Band | None, bool]:
    """(band, overlapping). Falls back to the band whose left edge is nearest."""
    for b in bands:
        if _overlaps(b.x0, b.x1, x0, x1):
            return b, True
    if not bands:
        return None, False
    return min(bands, key=lambda b: abs(b.x0 - x0)), False


def _group_by_band(cells: list[Cell], bands: list[_Band]) -> list[list[Cell]]:
    """Split a visual row into runs of cells that belong to the same band."""
    if len(bands) < 2:
        return [cells]
    groups: list[list[Cell]] = []
    last: _Band | None = None
    for cell in cells:
        band, _ = _band_for(cell[1], cell[2], bands)
        if groups and band is last:
            groups[-1].append(cell)
        else:
            groups.append([cell])
        last = band
    return groups


def _data_row(seg: _Segment, section: str, cfg: LayoutConfig) -> list[str]:
    cut = len(seg.cells) - seg.n_values
    head, values = seg.cells[:cut], seg.cells[cut:]
    unit = ""
    if len(head) >= 2 and len(head[-1]) <= cfg.max_unit_chars:
        head, unit = head[:-1], head[-1]
    label = clean_label(head)
    if section and label:
        label = f"{section} {label}"
    return [label, unit, *values]


def _flush(band: _Band, page: int) -> None:
    if not band.rows:
        return
    header = ["항목", "단위", *band.header]
    width = max(len(header), *(len(r) for r in band.rows))
    header += [f"col{i + 1}" for i in range(len(header), width)]
    rows = [r + [""] * (width - len(r)) for r in band.rows]
    band.tables.append(Table(page=page, index=len(band.tables), header=header, rows=rows,
                             title=band.title, kind="layout"))
    band.rows = []


def _keep(table: Table) -> bool:
    has_years = any(YEAR_RE.match(h) for h in table.header)
    return table.n_rows >= (1 if has_years else 3)


def extract_layout_tables_from_words(words: list[Word], page: int, cfg: LayoutConfig | None = None,
                                     index_start: int = 0) -> list[Table]:
    cfg = cfg or LayoutConfig()
    bands: list[_Band] = []
    for row in cluster_rows(words, cfg.row_tol):
        cells = split_cells(row, cfg.cell_gap)
        for group in _group_by_band(cells, bands):
            for seg in split_segments(group, cfg.min_values):
                band, overlapping = _band_for(seg.x0, seg.x1, bands)
                if seg.is_header:
                    if band is not None and overlapping:
                        _flush(band, page)
                        band.header, band.title, band.section = seg.years, seg.title, ""
                        band.grow(seg.x0, seg.x1)
                    else:
                        bands.append(_Band(header=seg.years, title=seg.title, x0=seg.x0, x1=seg.x1))
                elif seg.n_values >= cfg.min_values:
                    if band is None:
                        band = _Band(header=[], title="", x0=seg.x0, x1=seg.x1)
                        bands.append(band)
                    band.grow(seg.x0, seg.x1)
                    data_row = _data_row(seg, band.section, cfg)
                    if data_row[0]:
                        band.rows.append(data_row)
                elif band is not None and seg.n_values == 0:
                    label = clean_label(seg.cells)
                    if _is_section(label, cfg):
                        band.section = label
                        band.grow(seg.x0, seg.x1)
    tables: list[Table] = []
    for band in sorted(bands, key=lambda b: b.x0):
        _flush(band, page)
        tables.extend(t for t in band.tables if _keep(t))
    for k, t in enumerate(tables):
        t.index = index_start + k
    return tables


def page_words(page) -> list[Word]:
    """PyMuPDF page -> list of (x0, y0, x1, y1, text)."""
    return [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]

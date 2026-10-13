"""Load and validate data/eval_set.csv. The questions/answers are human-written."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ["id", "question", "answer", "gold_pages", "type", "split"]
TYPES = {"number", "table", "text", "multi"}
SPLITS = {"dev", "heldout"}
_ID_RE = re.compile(r"^q\d{3}$")


@dataclass
class EvalItem:
    id: str
    question: str
    answer: str
    gold_pages: list[int]
    type: str
    split: str


def parse_gold_pages(value: str) -> list[int]:
    pages = [p.strip() for p in value.split(";") if p.strip()]
    return [int(p) for p in pages]


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    """Return a list of human-readable problems; empty list means valid."""
    errors: list[str] = []
    seen: Counter[str] = Counter()
    for n, row in enumerate(rows, start=2):  # line 1 is the header
        rid = (row.get("id") or "").strip()
        where = f"line {n} ({rid or 'no id'})"
        if not _ID_RE.match(rid):
            errors.append(f"{where}: id must look like q001")
        seen[rid] += 1
        for col in ("question", "answer"):
            if not (row.get(col) or "").strip():
                errors.append(f"{where}: {col} is empty")
        try:
            pages = parse_gold_pages(row.get("gold_pages") or "")
            if not pages:
                errors.append(f"{where}: gold_pages is empty")
            if any(p <= 0 for p in pages):
                errors.append(f"{where}: gold_pages must be positive")
        except ValueError:
            errors.append(f"{where}: gold_pages must be integers separated by ';'")
        if (row.get("type") or "").strip() not in TYPES:
            errors.append(f"{where}: type must be one of {sorted(TYPES)}")
        if (row.get("split") or "").strip() not in SPLITS:
            errors.append(f"{where}: split must be one of {sorted(SPLITS)}")
    for rid, count in seen.items():
        if count > 1:
            errors.append(f"duplicate id {rid} ({count} times)")
    return errors


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if list(header) != COLUMNS:
            raise ValueError(f"header must be exactly {COLUMNS}, got {list(header)}")
        return list(reader)


def load_eval_set(path: str | Path, split: str | None = None) -> list[EvalItem]:
    """Load items. `split=None` returns everything; callers that tune must pass 'dev'."""
    rows = read_rows(path)
    errors = validate_rows(rows)
    if errors:
        raise ValueError("eval set is invalid:\n  " + "\n  ".join(errors))
    items = [
        EvalItem(
            id=r["id"].strip(),
            question=r["question"].strip(),
            answer=r["answer"].strip(),
            gold_pages=parse_gold_pages(r["gold_pages"]),
            type=r["type"].strip(),
            split=r["split"].strip(),
        )
        for r in rows
    ]
    if split is not None:
        items = [it for it in items if it.split == split]
    return items


def summarize(items: list[EvalItem]) -> dict[str, Counter[str]]:
    return {
        "split": Counter(it.split for it in items),
        "type": Counter(it.type for it in items),
    }

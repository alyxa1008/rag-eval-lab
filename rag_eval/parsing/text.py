"""Body-text extraction with PyMuPDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

_MULTI_BLANK = re.compile(r"\n\s*\n+")
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = _TRAILING_WS.sub("\n", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def extract_page_texts(pdf_path: str | Path) -> list[str]:
    """Return normalized body text for every page, index 0 == page 1."""
    with pymupdf.open(str(pdf_path)) as doc:
        return [normalize_text(page.get_text("text")) for page in doc]

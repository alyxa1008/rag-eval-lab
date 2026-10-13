"""Tokenizers for BM25. Korean needs morphemes: whitespace tokens keep particles attached."""

from __future__ import annotations

import re
from typing import Protocol

# Kiwi POS prefixes worth indexing: nouns, numerals, numbers, Latin, roots, verb/adjective stems
_KEEP = ("NN", "NR", "NP", "SN", "SL", "XR", "VV", "VA")
_WS = re.compile(r"[\w가-힣]+")


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[str]: ...


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WS.findall(text)]


class KiwiTokenizer:
    def __init__(self):
        from kiwipiepy import Kiwi

        self._kiwi = Kiwi()

    def tokenize(self, text: str) -> list[str]:
        return [t.form.lower() for t in self._kiwi.tokenize(text) if t.tag.startswith(_KEEP)]


def build_tokenizer(name: str) -> Tokenizer:
    if name == "kiwi":
        return KiwiTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    raise ValueError(f"unknown tokenizer: {name}")

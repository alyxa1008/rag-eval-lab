"""Embedding providers with a persistent on-disk cache (re-runs make zero API calls)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 64):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()
        self.model = model
        self.batch_size = batch_size
        self._client = OpenAI()   # reads OPENAI_API_KEY from the environment / .env
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            self.calls += 1
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out


class LocalEmbedder:
    """sentence-transformers model run locally (e.g. BAAI/bge-m3); no API key, no cost."""

    def __init__(self, model: str, batch_size: int = 16, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.batch_size = batch_size
        self._st = SentenceTransformer(model, device=device)
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vecs = self._st.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=len(texts) > 50)
        return vecs.tolist()


def build_embedder(provider: str, model: str, batch_size: int) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model, batch_size)
    if provider == "local":
        return LocalEmbedder(model, batch_size)
    raise ValueError(f"unknown embedding provider: {provider}")


class CachedEmbedder:
    """Wraps an Embedder; stores vectors in SQLite keyed by sha256(model + text)."""

    def __init__(self, inner: Embedder, cache_dir: str | Path):
        self.inner = inner
        self.model = inner.model
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in inner.model)
        self._db = sqlite3.connect(Path(cache_dir) / f"{safe}.sqlite")
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT NOT NULL)")
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\x00{text}".encode("utf-8")).hexdigest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(part))})", part).fetchall()
            found.update({k: json.loads(v) for k, v in rows})
        missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
        # de-duplicate identical texts before calling the API
        unique: dict[str, str] = {}
        for k, t in missing:
            unique.setdefault(k, t)
        self.hits += len(texts) - len(missing)
        self.misses += len(unique)
        if unique:
            vecs = self.inner.embed(list(unique.values()))
            with self._db:
                self._db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                                     [(k, json.dumps(v)) for k, v in zip(unique.keys(), vecs)])
            found.update(zip(unique.keys(), vecs))
        return [found[k] for k in keys]

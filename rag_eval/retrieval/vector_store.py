"""Thin ChromaDB wrapper: ids + embeddings in, (id, cosine similarity) out."""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from rag_eval.chunking import Chunk


def _collection_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-") or "chunks"
    return f"chunks_{safe}"[:512]


class VectorStore:
    def __init__(self, name: str, path: str | Path | None = None):
        self._client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection(
            _collection_name(name), metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self._col.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 256) -> None:
        for i in range(0, len(chunks), batch_size):
            part = chunks[i:i + batch_size]
            self._col.upsert(
                ids=[c.id for c in part],
                embeddings=embeddings[i:i + batch_size],
                documents=[c.text for c in part],
                metadatas=[{"page": c.page, "type": c.type, "source": c.source} for c in part],
            )

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        k = min(k, self.count())
        if k == 0:
            return []
        res = self._col.query(query_embeddings=[embedding], n_results=k, include=["distances"])
        return [(cid, 1.0 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

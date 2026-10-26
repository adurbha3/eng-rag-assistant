"""A small in-memory vector index: sentence-transformer embeddings plus cosine
similarity search over them (the same idea a vector database like FAISS or
Pinecone implements at larger scale)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from engrag.chunking import Chunk, build_chunks

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """texts -> (n, dim) L2-normalized embedding matrix."""
    embedder = _load_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorIndex:
    """An in-memory embedding index over a fixed set of chunks."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectors = embed_texts([c.text for c in chunks]) if chunks else np.zeros((0, 384))

    @classmethod
    def from_corpus(cls) -> "VectorIndex":
        return cls(build_chunks())

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        query_vec = embed_texts([query])[0]
        scores = self.vectors @ query_vec  # cosine similarity, vectors are L2-normalized
        top_indices = np.argsort(-scores)[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in top_indices]

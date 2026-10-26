"""Split engineering documents into sentence-level chunks for retrieval.

Sentence-level chunking gives finer-grained retrieval than whole-document
retrieval: a query can match the one or two sentences that actually answer it,
rather than pulling in an entire document as context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from engrag.corpus import Document, DOCUMENTS

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str


def split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return sentences


def chunk_document(doc: Document) -> list[Chunk]:
    sentences = split_sentences(doc.text)
    return [
        Chunk(chunk_id=f"{doc.doc_id}::{i}", doc_id=doc.doc_id, doc_title=doc.title, text=sentence)
        for i, sentence in enumerate(sentences)
    ]


def build_chunks(documents: list[Document] = DOCUMENTS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc))
    return chunks

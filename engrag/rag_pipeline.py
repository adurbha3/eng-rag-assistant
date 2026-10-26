"""Ties retrieval and generation together into a single retrieval-augmented
generation query function."""
from __future__ import annotations

import time
from dataclasses import dataclass

from engrag.embeddings import VectorIndex, ScoredChunk
from engrag.llm import LLMBackend

PROMPT_TEMPLATE = (
    "Answer the question using only the information in the context below. "
    "If the context does not contain the answer, say you do not know. "
    "Keep the answer to one or two sentences.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved: list[ScoredChunk]
    retrieval_seconds: float
    generation_seconds: float

    @property
    def source_doc_ids(self) -> list[str]:
        seen = []
        for scored in self.retrieved:
            if scored.chunk.doc_id not in seen:
                seen.append(scored.chunk.doc_id)
        return seen


def answer_question(
    question: str, index: VectorIndex, llm: LLMBackend, top_k: int = 4
) -> RAGResult:
    t0 = time.perf_counter()
    retrieved = index.search(question, top_k=top_k)
    retrieval_seconds = time.perf_counter() - t0

    context = " ".join(scored.chunk.text for scored in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    t1 = time.perf_counter()
    answer = llm.generate(prompt)
    generation_seconds = time.perf_counter() - t1

    return RAGResult(
        question=question,
        answer=answer,
        retrieved=retrieved,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
    )

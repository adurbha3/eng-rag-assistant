#!/usr/bin/env python
"""Ask the RAG assistant a single question from the command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engrag.embeddings import VectorIndex
from engrag.llm import LocalLLM
from engrag.rag_pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", type=str, help="Question to ask the engineering knowledge base")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    print("Loading embedding index and local model (first run downloads ~400MB)...")
    index = VectorIndex.from_corpus()
    llm = LocalLLM()

    result = answer_question(args.question, index, llm, top_k=args.top_k)

    print(f"\nQ: {result.question}")
    print(f"A: {result.answer}")
    print(f"\nSources: {', '.join(result.source_doc_ids)}")
    print(f"Retrieval: {result.retrieval_seconds * 1000:.0f} ms | Generation: {result.generation_seconds * 1000:.0f} ms")


if __name__ == "__main__":
    main()

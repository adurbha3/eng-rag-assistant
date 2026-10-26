#!/usr/bin/env python
"""Run the eval suite against the RAG pipeline and print retrieval, answer
quality, and latency metrics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engrag.embeddings import VectorIndex
from engrag.llm import LocalLLM
from engrag.evals import run_evals, EVAL_SET


def main() -> None:
    print(f"Loading embedding index and local model, then running {len(EVAL_SET)} eval cases...")
    index = VectorIndex.from_corpus()
    llm = LocalLLM()

    report = run_evals(index, llm)

    print(f"\nEval report over {report.n_cases} cases:")
    print(f"  Retrieval recall@4:        {report.recall_at_k:.3f}")
    print(f"  Retrieval mean reciprocal rank: {report.mean_reciprocal_rank:.3f}")
    print(f"  Answer keyword coverage:   {report.keyword_coverage:.3f}")
    print(f"  Mean retrieval latency:    {report.mean_retrieval_seconds * 1000:.1f} ms")
    print(f"  Mean generation latency:   {report.mean_generation_seconds * 1000:.1f} ms")


if __name__ == "__main__":
    main()

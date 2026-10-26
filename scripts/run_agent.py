#!/usr/bin/env python
"""Run the tool-using agent on a few example queries: a calculation and
document-search questions, to show the routing decision in action."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engrag.agent import EngineeringAgent
from engrag.embeddings import VectorIndex
from engrag.llm import LocalLLM

EXAMPLE_QUERIES = [
    "480 * 520",
    "What is a dual active bridge converter used for?",
    "(250000 / 480)",
    "Why do data centers need new power infrastructure for AI training?",
]


def main() -> None:
    print("Loading embedding index and local model (first run downloads ~400MB)...")
    index = VectorIndex.from_corpus()
    llm = LocalLLM()
    agent = EngineeringAgent(index=index, llm=llm)

    for query in EXAMPLE_QUERIES:
        result = agent.run(query)
        print(f"\nQuery: {query}")
        print(f"  Routed to: {result.tool.value}")
        print(f"  Output: {result.output}")


if __name__ == "__main__":
    main()

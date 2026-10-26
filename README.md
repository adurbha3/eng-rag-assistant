# EngDocs RAG Assistant

**A retrieval-augmented engineering knowledge assistant with a small tool-using agent, running entirely on local open-source models — no API key required — plus a built-in evaluation harness.**

## Why this exists

This project targets a specific kind of role directly: building AI frameworks, agentic workflows, and knowledge assistants that help engineering teams access and utilize engineering data, fine-tuning and evaluating LLMs, and developing RAG applications over documentation. Rather than wrap a hosted API, I built the whole pipeline — embeddings, retrieval, generation, agent routing, and evaluation — against small open-source models that run locally on CPU, which forced real engineering tradeoffs instead of hiding behind a large hosted model's capability.

The knowledge base is a set of original short documents (written for this project, not copied from any source) covering power-electronics and power-infrastructure topics — solid-state transformers, wide-bandgap semiconductors, medium-voltage conversion, EV fast charging, data center power, grid interconnection, and more — the same kind of domain-specific engineering documentation a RAG assistant for a power-infrastructure company would be built on.

## Features

- **Retrieval** (`engrag/embeddings.py`) — sentence-level chunking (`engrag/chunking.py`) and a `sentence-transformers/all-MiniLM-L6-v2` embedding index with cosine-similarity search, the same mechanism a vector database implements at larger scale.
- **Generation** (`engrag/llm.py`) — a pluggable `LLMBackend` interface: `LocalLLM` runs `HuggingFaceTB/SmolLM2-360M-Instruct` fully locally for real generation, and a deterministic `StubLLM` lets the rest of the pipeline be unit-tested in milliseconds without loading a model. Swapping in a hosted model later means implementing one method.
- **RAG pipeline** (`engrag/rag_pipeline.py`) — retrieves relevant chunks, builds a grounded prompt, generates an answer, and reports which source documents it cited plus retrieval/generation latency.
- **Tool-using agent** (`engrag/agent.py`, `engrag/tools.py`) — classifies whether a query needs a calculation or a document lookup, then calls a safe AST-based arithmetic evaluator or the RAG pipeline accordingly. The routing logic is deliberately scoped to one classification decision, a choice made after testing what a 360M-parameter model can reliably do, not a shortcut taken unknowingly.
- **Evaluation harness** (`engrag/evals.py`) — a hand-written 20-question eval set with expected source documents and answer keywords, scored by retrieval recall@k, mean reciprocal rank, answer keyword coverage, and end-to-end latency.
- **Tested** — 24 automated tests (`pytest`), almost all running against the real embedding model or a fast deterministic LLM stub so the suite runs in seconds, plus one true end-to-end integration test against the real local language model — run automatically on push via GitHub Actions.

## Install

```bash
git clone https://github.com/adurbha3/eng-rag-assistant.git
cd eng-rag-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Needs Python 3.9+. The first run downloads the embedding model (~90MB) and the local LLM (~360M parameters, ~700MB) from Hugging Face.

## Usage

Ask the assistant a question directly:

```bash
python scripts/ask.py "Why are SiC devices used in power converters?"
```

Run the tool-using agent on a mix of calculation and lookup queries:

```bash
python scripts/run_agent.py
```

Run the evaluation harness:

```bash
python scripts/run_evals.py
```

## Project layout

```
engrag/
  corpus.py        original engineering-documentation knowledge base
  chunking.py       sentence-level document chunking
  embeddings.py      sentence-transformer embeddings + in-memory vector search
  llm.py            pluggable LLM backend: LocalLLM (real) and StubLLM (fast/deterministic)
  rag_pipeline.py    retrieval + generation, with source attribution and latency
  tools.py          safe arithmetic calculator, RAG-backed document search
  agent.py          classifies a query and routes it to the right tool
  evals.py          retrieval and answer-quality evaluation harness
scripts/
  ask.py            CLI: ask one question
  run_agent.py      CLI: run the agent on example queries
  run_evals.py      CLI: run the eval suite and print metrics
tests/              pytest suite covering every module above
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

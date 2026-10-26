"""An evaluation harness for the RAG pipeline: retrieval quality (recall@k,
mean reciprocal rank), answer quality (keyword overlap against a reference
answer), and latency -- the kind of eval suite a production RAG system is
checked against before and after a change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from engrag.embeddings import VectorIndex
from engrag.llm import LLMBackend
from engrag.rag_pipeline import answer_question


@dataclass
class EvalCase:
    question: str
    expected_doc_id: str
    answer_keywords: list[str]  # any of these appearing in the answer counts as "covered"


EVAL_SET: list[EvalCase] = [
    EvalCase("What is a solid-state transformer?", "sst-overview", ["power electronic", "converter", "compact"]),
    EvalCase("Why are SiC devices used in power converters?", "sic-semiconductors", ["switch", "fast", "temperature", "loss"]),
    EvalCase("What is gallium nitride used for?", "gan-semiconductors", ["charger", "compact", "fast"]),
    EvalCase("What voltage range counts as medium voltage?", "mv-power-conversion", ["1 kv", "35 kv", "kv"]),
    EvalCase("How does DC fast charging work?", "ev-fast-charging", ["convert", "dc", "battery"]),
    EvalCase("Why is data center power infrastructure changing?", "data-center-power", ["ai", "density", "rack"]),
    EvalCase("What does a battery energy storage system do?", "battery-storage", ["charg", "discharg", "grid"]),
    EvalCase("What do grid interconnection standards cover?", "grid-interconnection", ["ride-through", "harmonic", "fault"]),
    EvalCase("What is a dual active bridge used for?", "converter-topologies", ["bidirectional", "isolat", "dc-dc"]),
    EvalCase("How is heat removed from power electronics?", "thermal-management", ["heatsink", "cooling", "liquid"]),
    EvalCase("What does power factor correction do?", "power-factor-correction", ["phase", "harmonic", "current"]),
    EvalCase("What is the tradeoff of raising switching frequency?", "switching-losses", ["loss", "smaller", "magnetic"]),
    EvalCase("Why is galvanic isolation needed?", "galvanic-isolation", ["safety", "ground", "separat"]),
    EvalCase("What causes transformer core losses?", "transformer-efficiency", ["magnetiz", "frequency", "flux"]),
    EvalCase("What does a solar inverter do?", "solar-inverters", ["photovoltaic", "dc", "ac", "maximum power"]),
    EvalCase("How fast must power electronics respond to a fault?", "fault-protection", ["microsecond", "overcurrent", "short circuit"]),
    EvalCase("What is a modular multilevel converter?", "modular-multilevel", ["submodule", "series", "voltage step"]),
    EvalCase("Why has digital control replaced analog control?", "converter-controls", ["microcontroller", "dsp", "algorithm"]),
    EvalCase("What is EMI in power electronics?", "emi-emc", ["interference", "filter", "noise"]),
    EvalCase("How is reliability achieved in critical power infrastructure?", "reliability-redundancy", ["redundant", "failure", "replacement"]),
]


@dataclass
class EvalReport:
    n_cases: int
    recall_at_k: float
    mean_reciprocal_rank: float
    keyword_coverage: float
    mean_retrieval_seconds: float
    mean_generation_seconds: float


def _reciprocal_rank(expected_doc_id: str, retrieved_doc_ids: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0


def _keyword_covered(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(re.search(re.escape(kw.lower()), answer_lower) for kw in keywords)


def run_evals(index: VectorIndex, llm: LLMBackend, top_k: int = 4, cases: list[EvalCase] = EVAL_SET) -> EvalReport:
    reciprocal_ranks = []
    hit_at_k = []
    keyword_hits = []
    retrieval_times = []
    generation_times = []

    for case in cases:
        result = answer_question(case.question, index, llm, top_k=top_k)
        retrieved_doc_ids = result.source_doc_ids

        reciprocal_ranks.append(_reciprocal_rank(case.expected_doc_id, retrieved_doc_ids))
        hit_at_k.append(case.expected_doc_id in retrieved_doc_ids)
        keyword_hits.append(_keyword_covered(result.answer, case.answer_keywords))
        retrieval_times.append(result.retrieval_seconds)
        generation_times.append(result.generation_seconds)

    n = len(cases)
    return EvalReport(
        n_cases=n,
        recall_at_k=sum(hit_at_k) / n,
        mean_reciprocal_rank=sum(reciprocal_ranks) / n,
        keyword_coverage=sum(keyword_hits) / n,
        mean_retrieval_seconds=sum(retrieval_times) / n,
        mean_generation_seconds=sum(generation_times) / n,
    )

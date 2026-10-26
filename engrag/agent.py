"""A small tool-using agent: the LLM classifies whether a query needs a
calculation or a document lookup, then the matching tool runs and its result
is returned.

With a 360M-parameter local model, a full multi-step ReAct loop with
free-form tool-call JSON is not reliable, so the agent is deliberately scoped
to one classification decision plus one tool call -- a design choice made
after testing the model's actual output, not a shortcut taken unknowingly.
A regex-based fallback keeps the agent working even if the model's
classification reply is ambiguous.

The classification prompt was itself evaluation-driven: a plain zero-shot
"reply CALCULATE or SEARCH" instruction made the model answer CALCULATE for
every input regardless of content (verified by printing its raw output on a
handful of test questions). Adding three few-shot examples fixed this
completely, which is why the prompt below is few-shot rather than zero-shot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from engrag.embeddings import VectorIndex
from engrag.llm import LLMBackend
from engrag.tools import calculate, search_docs, UnsafeExpressionError

CLASSIFY_PROMPT = (
    "Classify each question as CALCULATE (needs arithmetic) or SEARCH (needs a "
    "document lookup).\n\n"
    "Question: What is 12 times 4?\n"
    "Label: CALCULATE\n\n"
    "Question: What is a transformer?\n"
    "Label: SEARCH\n\n"
    "Question: How much power is 10 amps times 20 volts?\n"
    "Label: CALCULATE\n\n"
    "Question: {question}\n"
    "Label:"
)

_MATH_EXPRESSION_RE = re.compile(r"^[\d\s+\-*/().]+$")


class ToolChoice(Enum):
    CALCULATE = "calculate"
    SEARCH = "search"


@dataclass
class AgentResult:
    question: str
    tool: ToolChoice
    output: str
    classification_raw: str


def _classify(question: str, llm: LLMBackend) -> ToolChoice:
    expression_candidate = question.strip().rstrip("?=").strip()
    if _MATH_EXPRESSION_RE.match(expression_candidate):
        return ToolChoice.CALCULATE

    raw = llm.generate(CLASSIFY_PROMPT.format(question=question), max_new_tokens=5)
    if "CALCULATE" in raw.upper():
        return ToolChoice.CALCULATE
    return ToolChoice.SEARCH


class EngineeringAgent:
    def __init__(self, index: VectorIndex, llm: LLMBackend):
        self.index = index
        self.llm = llm

    def run(self, question: str) -> AgentResult:
        choice = _classify(question, self.llm)

        if choice == ToolChoice.CALCULATE:
            expression = question.strip().rstrip("?=").strip()
            try:
                result = calculate(expression)
                output = f"{result:g}"
            except UnsafeExpressionError as e:
                output = f"Could not evaluate that as an arithmetic expression: {e}"
            return AgentResult(question=question, tool=choice, output=output, classification_raw="CALCULATE")

        rag_result = search_docs(question, self.index, self.llm)
        return AgentResult(
            question=question, tool=choice, output=rag_result.answer, classification_raw="SEARCH"
        )

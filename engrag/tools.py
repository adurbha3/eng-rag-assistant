"""Tools the agent can call: a safe arithmetic calculator (e.g. power = voltage
x current) and a RAG-backed document search."""
from __future__ import annotations

import ast
import operator

from engrag.embeddings import VectorIndex
from engrag.llm import LLMBackend
from engrag.rag_pipeline import RAGResult, answer_question

_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise UnsafeExpressionError(f"expression contains a disallowed element: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """Safely evaluate a numeric expression (+, -, *, /, **) with no name
    lookups or function calls, unlike a bare eval()."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"could not parse expression: {expression!r}") from e
    return _eval_node(tree.body)


def search_docs(query: str, index: VectorIndex, llm: LLMBackend, top_k: int = 4) -> RAGResult:
    return answer_question(query, index, llm, top_k=top_k)

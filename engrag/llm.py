"""A small pluggable LLM backend interface, so the RAG pipeline and agent can
run against a real local model for actual use, or a fast deterministic stub
for unit tests -- the same dependency-injection pattern used to swap in a
hosted model (OpenAI, Anthropic, etc.) in production without touching the
RAG or agent logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

LOCAL_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """Return a generated response to a single-turn prompt."""


@lru_cache(maxsize=1)
def _load_local_pipeline():
    from transformers import pipeline

    return pipeline("text-generation", model=LOCAL_MODEL_NAME)


class LocalLLM(LLMBackend):
    """A small open-source instruction-tuned model running fully locally (no API key)."""

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        pipe = _load_local_pipeline()
        messages = [{"role": "user", "content": prompt}]
        output = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
        return output[0]["generated_text"][-1]["content"].strip()


class StubLLM(LLMBackend):
    """A deterministic, instant backend for unit tests: echoes the most
    relevant-looking sentence from the prompt's context section instead of
    actually generating, so tests never need to load a model."""

    def generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        if "Context:" in prompt:
            context = prompt.split("Context:", 1)[1]
            context = context.split("Question:")[0].strip()
            first_sentence = context.split(".")[0].strip()
            return first_sentence + "." if first_sentence else "I do not know."
        return "I do not know."

"""One true end-to-end test using the real local model (not the stub) to
confirm the full RAG pipeline actually produces a grounded answer, not just
that the plumbing between mocked pieces is correct. Skipped if the model
cannot be downloaded (e.g. no network in a restricted CI runner)."""
import pytest

from engrag.chunking import Chunk
from engrag.embeddings import VectorIndex
from engrag.rag_pipeline import answer_question


@pytest.fixture(scope="module")
def real_llm():
    try:
        from engrag.llm import LocalLLM

        llm = LocalLLM()
        llm.generate("Say hi.", max_new_tokens=5)  # force the model to load now
        return llm
    except Exception as e:
        pytest.skip(f"local model unavailable in this environment: {e}")


def test_real_local_model_answers_grounded_question(real_llm):
    chunks = [
        Chunk("v::0", "v", "Voltage", "The rated output voltage of this converter is 480 volts."),
    ]
    index = VectorIndex(chunks)
    result = answer_question("What is the rated output voltage?", index, real_llm, top_k=1)
    assert "480" in result.answer

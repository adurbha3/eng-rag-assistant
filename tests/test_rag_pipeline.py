from engrag.chunking import Chunk
from engrag.embeddings import VectorIndex
from engrag.llm import StubLLM
from engrag.rag_pipeline import answer_question


def _toy_index():
    chunks = [
        Chunk("sic::0", "sic", "SiC", "Silicon carbide transistors switch faster than silicon transistors."),
        Chunk("cook::0", "cook", "Cooking", "Bread needs flour, water, and yeast."),
    ]
    return VectorIndex(chunks)


def test_answer_question_retrieves_and_cites_relevant_source():
    index = _toy_index()
    llm = StubLLM()
    result = answer_question("What switches faster than silicon?", index, llm, top_k=1)
    assert result.source_doc_ids == ["sic"]
    assert "silicon carbide" in result.answer.lower()


def test_answer_question_tracks_timing():
    index = _toy_index()
    llm = StubLLM()
    result = answer_question("anything", index, llm, top_k=2)
    assert result.retrieval_seconds >= 0
    assert result.generation_seconds >= 0


def test_source_doc_ids_deduplicates_in_order():
    index = _toy_index()
    llm = StubLLM()
    result = answer_question("silicon", index, llm, top_k=2)
    assert len(result.source_doc_ids) == len(set(result.source_doc_ids))

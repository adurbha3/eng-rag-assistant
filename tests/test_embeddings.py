import numpy as np

from engrag.chunking import Chunk
from engrag.embeddings import VectorIndex, embed_texts


def test_embed_texts_returns_normalized_vectors():
    vectors = embed_texts(["hello world", "power electronics"])
    assert vectors.shape[0] == 2
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_vector_index_search_ranks_relevant_chunk_first():
    chunks = [
        Chunk("a::0", "a", "Cooking", "A recipe for baking bread requires flour, water, and yeast."),
        Chunk("b::0", "b", "Power", "Silicon carbide transistors switch faster than silicon transistors."),
        Chunk("c::0", "c", "Sports", "The basketball team won the championship game last night."),
    ]
    index = VectorIndex(chunks)
    results = index.search("What semiconductor material switches quickly?", top_k=1)
    assert results[0].chunk.doc_id == "b"


def test_vector_index_search_respects_top_k():
    chunks = [
        Chunk(f"d{i}::0", f"d{i}", f"Doc {i}", f"This is document number {i} about topic {i}.")
        for i in range(10)
    ]
    index = VectorIndex(chunks)
    results = index.search("topic 5", top_k=3)
    assert len(results) == 3


def test_empty_index_returns_no_results():
    index = VectorIndex([])
    assert index.search("anything") == []

from engrag.embeddings import VectorIndex
from engrag.llm import StubLLM
from engrag.evals import run_evals, EvalCase, _reciprocal_rank, _keyword_covered


def test_reciprocal_rank_scoring():
    assert _reciprocal_rank("a", ["a", "b", "c"]) == 1.0
    assert _reciprocal_rank("b", ["a", "b", "c"]) == 0.5
    assert _reciprocal_rank("z", ["a", "b", "c"]) == 0.0


def test_keyword_covered_is_case_insensitive():
    assert _keyword_covered("Silicon Carbide is fast", ["carbide"])
    assert not _keyword_covered("Nothing relevant here", ["carbide"])


def test_run_evals_against_real_corpus_with_stub_llm():
    """Uses the real embedding model and corpus (real retrieval quality check)
    but the fast StubLLM, so this test runs in a couple of seconds, not the
    ~20s a full local-model load would take."""
    index = VectorIndex.from_corpus()
    llm = StubLLM()
    cases = [
        EvalCase("What is a solid-state transformer?", "sst-overview", ["power electronic", "converter", "compact"]),
        EvalCase("Why are SiC devices used in power converters?", "sic-semiconductors", ["switch", "fast"]),
    ]
    report = run_evals(index, llm, top_k=4, cases=cases)
    assert report.n_cases == 2
    assert 0.0 <= report.recall_at_k <= 1.0
    assert 0.0 <= report.mean_reciprocal_rank <= 1.0
    assert report.recall_at_k >= 0.5  # retrieval should find at least one of these two easily

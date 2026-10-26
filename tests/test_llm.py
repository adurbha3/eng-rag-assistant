from engrag.llm import StubLLM


def test_stub_llm_extracts_first_context_sentence():
    llm = StubLLM()
    prompt = (
        "Answer using only the context.\n\n"
        "Context: SiC devices switch faster than silicon. They also run hotter.\n\n"
        "Question: How fast do SiC devices switch?\nAnswer:"
    )
    answer = llm.generate(prompt)
    assert answer == "SiC devices switch faster than silicon."


def test_stub_llm_handles_missing_context():
    llm = StubLLM()
    assert llm.generate("Question: anything?\nAnswer:") == "I do not know."

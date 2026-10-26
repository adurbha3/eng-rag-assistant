from engrag.chunking import split_sentences, chunk_document, build_chunks
from engrag.corpus import Document, DOCUMENTS


def test_split_sentences_basic():
    sentences = split_sentences("This is one. This is two! Is this three?")
    assert sentences == ["This is one.", "This is two!", "Is this three?"]


def test_chunk_document_preserves_doc_metadata():
    doc = Document("test-doc", "Test Title", "Sentence one. Sentence two.")
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert all(c.doc_id == "test-doc" for c in chunks)
    assert all(c.doc_title == "Test Title" for c in chunks)
    assert chunks[0].chunk_id == "test-doc::0"
    assert chunks[1].text == "Sentence two."


def test_build_chunks_covers_every_document():
    chunks = build_chunks()
    doc_ids_in_chunks = {c.doc_id for c in chunks}
    assert doc_ids_in_chunks == {doc.doc_id for doc in DOCUMENTS}
    assert len(chunks) >= len(DOCUMENTS)

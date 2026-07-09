import pytest
from src.python.insight_rag.ingestion.document_loader import Document
from src.python.insight_rag.ingestion.chunker import RecursiveChunker

def test_chunker_split():
    doc = Document(
        text="This is a long piece of text representing some documentation blocks. It repeats words to fill up block boundaries.",
        metadata={"source": "test.txt", "page": 1}
    )
    chunker = RecursiveChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.split_documents([doc])
    
    assert len(chunks) > 0
    assert chunks[0].metadata["source"] == "test.txt"
    assert "chunk_id" in chunks[0].metadata

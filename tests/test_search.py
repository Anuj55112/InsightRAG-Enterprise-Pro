import pytest
from src.python.insight_rag.ingestion.document_loader import Document
from src.python.insight_rag.search.bm25_search import BM25Searcher
from src.python.insight_rag.search.embedder import SentenceEmbedder
from src.python.insight_rag.search.hybrid_search import HybridSearcher

def test_bm25_searcher():
    doc1 = Document(text="Machine learning models are trained on large datasets.", metadata={"source": "doc1.txt"})
    doc2 = Document(text="Attention mechanisms map sequences in parallel.", metadata={"source": "doc2.txt"})
    
    searcher = BM25Searcher()
    searcher.fit([doc1, doc2])
    
    results = searcher.search("attention mechanisms", top_k=1)
    assert len(results) > 0
    assert "attention" in results[0][0].text.lower()

def test_hybrid_searcher():
    doc1 = Document(text="dense vectors encode semantic meaning.", metadata={"source": "doc1.txt", "chunk_id": "c1"})
    doc2 = Document(text="sparse term frequency scores document keywords.", metadata={"source": "doc2.txt", "chunk_id": "c2"})
    
    embedder = SentenceEmbedder()
    hybrid = HybridSearcher(embedder)
    
    hybrid.fit([doc1, doc2])
    results = hybrid.search("semantic meaning", top_k=1)
    
    assert len(results) > 0

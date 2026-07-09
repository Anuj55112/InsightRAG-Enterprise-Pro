import numpy as np
from typing import List, Dict, Tuple, Any

from src.python.insight_rag.ingestion.document_loader import Document
from src.python.insight_rag.search.bm25_search import BM25Searcher
from src.python.insight_rag.search.embedder import SentenceEmbedder

class HybridSearcher:
    """Combines BM25 lexical search with dense vector searches using Reciprocal Rank Fusion (RRF)."""
    def __init__(self, embedder: SentenceEmbedder, rrf_k: int = 60):
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.bm25 = BM25Searcher()
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []

    def fit(self, documents: List[Document]):
        """Indexes documents and generates dense vectors."""
        self.documents = documents
        if not documents:
            return
            
        # Fit BM25
        self.bm25.fit(documents)
        
        # Generate dense embeddings
        texts = [doc.text for doc in documents]
        self.embeddings = list(self.embedder.encode(texts))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Executes lexical and semantic queries, fusion-ranking them using RRF."""
        if not self.documents:
            return []
            
        # 1. Lexical BM25 search
        bm25_results = self.bm25.search(query, top_k=len(self.documents))
        
        # 2. Dense Vector search
        query_emb = self.embedder.encode([query])[0]
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        
        dense_scores = []
        for emb in self.embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
            dense_scores.append(float(np.dot(emb_norm, query_norm)))
            
        dense_ranked_indices = np.argsort(dense_scores)[::-1]
        dense_results = [(self.documents[idx], dense_scores[idx]) for idx in dense_ranked_indices]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # Create map of text chunk to RRF score
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Helper to apply rank score
        def apply_rrf(results_list):
            for rank, (doc, _) in enumerate(results_list, 1):
                chunk_id = doc.metadata.get("chunk_id", doc.text[:50])
                doc_map[chunk_id] = doc
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
                
        apply_rrf(bm25_results)
        apply_rrf(dense_results)
        
        # Sort chunks by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [(doc_map[chunk_id], score) for chunk_id, score in sorted_chunks]

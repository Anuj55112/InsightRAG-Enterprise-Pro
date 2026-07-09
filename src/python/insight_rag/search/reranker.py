import numpy as np
from typing import List, Tuple, Any
from src.python.insight_rag.ingestion.document_loader import Document

class CrossEncoderReranker:
    """Uses a fine-grained Cross-Encoder model to score and re-rank query-document pairs."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            print(f"Loaded cross-encoder: {self.model_name}")
        except Exception as e:
            print(f"Could not load CrossEncoder: {e}. Using fallback TF-IDF similarity reranker.")

    def rerank(self, query: str, candidates: List[Document]) -> List[Tuple[Document, float]]:
        """Reranks candidates based on query relevance scores."""
        if not candidates:
            return []
            
        if self.model is not None:
            try:
                pairs = [[query, doc.text] for doc in candidates]
                scores = self.model.predict(pairs)
                # Sort descending
                ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                return [(doc, float(score)) for doc, score in ranked_results]
            except Exception as e:
                print(f"Cross-encoder reranking failed: {e}. Falling back.")
                
        # Heuristic fallback: Jaccard overlap scorer
        scored_results = []
        query_words = set(query.lower().split())
        for doc in candidates:
            doc_words = set(doc.text.lower().split())
            intersection = query_words.intersection(doc_words)
            union = query_words.union(doc_words)
            score = len(intersection) / len(union) if union else 0.0
            scored_results.append((doc, float(score)))
            
        return sorted(scored_results, key=lambda x: x[1], reverse=True)

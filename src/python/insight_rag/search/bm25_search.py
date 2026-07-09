import math
import re
from typing import List, Dict, Tuple, Any
from src.python.insight_rag.ingestion.document_loader import Document

class BM25Searcher:
    """Mathematical BM25 sparse lexical search engine."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Simple lowercase tokenizer with token bounds
        return re.findall(r'\b\w{3,}\b', text.lower())

    def fit(self, documents: List[Document]):
        """Fits sparse frequency index over document collection."""
        self.documents = documents
        self.corpus_size = len(documents)
        
        if self.corpus_size == 0:
            return
            
        self.doc_lens = []
        self.doc_term_freqs = []
        
        # Word DF counter
        df: Dict[str, int] = {}
        
        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc.text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            tf: Dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self.doc_term_freqs.append(tf)
            
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0
        
        # Compute IDF
        for word, word_df in df.items():
            # BM25 standard IDF with smoothing
            self.idf[word] = math.log((self.corpus_size - word_df + 0.5) / (word_df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Returns documents ranked by BM25 scoring match."""
        if self.corpus_size == 0:
            return []
            
        query_tokens = self._tokenize(query)
        scores = []
        
        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lens[idx]
            tf_dict = self.doc_term_freqs[idx]
            
            for token in query_tokens:
                if token in tf_dict:
                    tf = tf_dict[token]
                    token_idf = self.idf.get(token, 0.0)
                    
                    # BM25 score formula
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += token_idf * (numerator / denominator)
            scores.append(score)
            
        # Rank
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]

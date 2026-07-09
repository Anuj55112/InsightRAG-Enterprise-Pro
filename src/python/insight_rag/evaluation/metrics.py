import re
from typing import Dict, List, Set

class RAGEvaluator:
    """Calculates context relevance, faithfulness, and hallucination scores mathematically."""
    
    def _get_words(self, text: str) -> Set[str]:
        return set(re.findall(r'\b\w{3,}\b', text.lower()))

    def score_context_relevance(self, query: str, context: str) -> float:
        """Measures lexical overlap between query tokens and retrieved context."""
        query_words = self._get_words(query)
        context_words = self._get_words(context)
        
        if not query_words:
            return 1.0
            
        overlap = query_words.intersection(context_words)
        return float(len(overlap) / len(query_words))

    def score_faithfulness(self, answer: str, context: str) -> float:
        """Measures if statements in the answer are grounded in the retrieved context."""
        answer_words = self._get_words(answer)
        context_words = self._get_words(context)
        
        if not answer_words:
            return 1.0
            
        grounded = answer_words.intersection(context_words)
        return float(len(grounded) / len(answer_words))

    def score_correctness(self, answer: str, reference: str) -> float:
        """Measures token similarity between candidate answers and reference targets."""
        ans_words = self._get_words(answer)
        ref_words = self._get_words(reference)
        
        if not ans_words or not ref_words:
            return 0.0
            
        intersection = ans_words.intersection(ref_words)
        union = ans_words.union(ref_words)
        return float(len(intersection) / len(union))

    def score_hallucination(self, answer: str, context: str) -> float:
        """Measures claims made in the answer that do not appear in the context."""
        faithfulness = self.score_faithfulness(answer, context)
        # Hallucination is inverse of grounding/faithfulness
        return float(1.0 - faithfulness)

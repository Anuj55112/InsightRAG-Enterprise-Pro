import pytest
from src.python.insight_rag.evaluation.metrics import RAGEvaluator

def test_evaluation_metrics():
    evaluator = RAGEvaluator()
    
    query = "self-attention mechanism"
    context = "Self-attention mechanisms route input sequence values dynamically."
    answer = "Self-attention mechanism maps sequence elements."
    reference = "Self-attention mechanism maps sequence elements."
    
    relevance = evaluator.score_context_relevance(query, context)
    faithfulness = evaluator.score_faithfulness(answer, context)
    correctness = evaluator.score_correctness(answer, reference)
    hallucination = evaluator.score_hallucination(answer, context)
    
    assert relevance > 0.0
    assert faithfulness > 0.0
    assert correctness == 1.0
    assert hallucination == 1.0 - faithfulness

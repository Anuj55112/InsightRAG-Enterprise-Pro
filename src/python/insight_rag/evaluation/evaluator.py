import pandas as pd
from typing import List, Dict, Any

from src.python.insight_rag.agents.rag_agent import EnterpriseRAGAgent
from src.python.insight_rag.evaluation.metrics import RAGEvaluator

class RAGBenchEvaluator:
    """Executes evaluation suite over QA datasets."""
    def __init__(self, agent: EnterpriseRAGAgent):
        self.agent = agent
        self.evaluator = RAGEvaluator()

    def evaluate_dataset(self, test_set: List[Dict[str, str]]) -> Dict[str, Any]:
        """Runs evaluation over dataset containing queries and reference answers."""
        records = []
        
        sum_relevance = 0.0
        sum_faithfulness = 0.0
        sum_correctness = 0.0
        sum_hallucination = 0.0
        
        for qa in test_set:
            query = qa["query"]
            ref = qa.get("reference", "")
            
            # Answer
            res = self.agent.answer_query(query)
            ans = res["answer"]
            context = res["context"]
            
            # Score
            rel = self.evaluator.score_context_relevance(query, context)
            faith = self.evaluator.score_faithfulness(ans, context)
            corr = self.evaluator.score_correctness(ans, ref) if ref else 1.0
            halluc = self.evaluator.score_hallucination(ans, context)
            
            sum_relevance += rel
            sum_faithfulness += faith
            sum_correctness += corr
            sum_hallucination += halluc
            
            records.append({
                "query": query,
                "context_relevance": rel,
                "faithfulness": faith,
                "correctness": corr,
                "hallucination": halluc
            })
            
        count = len(test_set) if test_set else 1
        avg_rel = sum_relevance / count
        avg_faith = sum_faithfulness / count
        avg_corr = sum_correctness / count
        avg_halluc = sum_hallucination / count
        
        # Build report
        df = pd.DataFrame(records)
        print("\n" + "="*50)
        print("InsightRAG Pro Evaluation Summary Report")
        print("="*50)
        print(f"Context Relevance: {avg_rel:.4f}")
        print(f"Faithfulness:      {avg_faith:.4f}")
        print(f"Answer Correctness: {avg_corr:.4f}")
        print(f"Hallucination Rate: {avg_halluc:.4f}")
        print("="*50 + "\n")
        
        return {
            "records": records,
            "averages": {
                "context_relevance": avg_rel,
                "faithfulness": avg_faith,
                "correctness": avg_corr,
                "hallucination": avg_halluc
            }
        }

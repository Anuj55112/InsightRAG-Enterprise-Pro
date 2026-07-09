from typing import Dict, Any

class QueryRouter:
    """Classifies user queries to determine the optimal retrieval strategy."""
    
    def route(self, query: str) -> Dict[str, Any]:
        """Classifies query and returns routing metadata."""
        q = query.lower()
        
        # Heuristics for routing
        if any(w in q for w in ["how is", "relation", "connect", "influence", "network", "structure", "community"]):
            strategy = "graphrag"
            reason = "Query demands structural or relational analysis over the entity knowledge graph."
        elif any(w in q for w in ["compare", "versus", "vs", "difference", "contrast"]):
            strategy = "comparison"
            reason = "Query requests comparative breakdown over multiple document source points."
        elif any(w in q for w in ["what is", "define", "explain", "exact", "code"]):
            strategy = "vector_search"
            reason = "Query requests factual, specific content suited for dense hybrid retrieval."
        else:
            strategy = "hybrid"
            reason = "General query format suited for standard dense-sparse Reciprocal Rank Fusion retrieval."
            
        return {
            "query": query,
            "strategy": strategy,
            "reason": reason
        }

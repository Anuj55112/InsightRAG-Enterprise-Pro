from typing import List, Dict, Tuple, Any

from src.python.insight_rag.ingestion.document_loader import Document
from src.python.insight_rag.search.hybrid_search import HybridSearcher
from src.python.insight_rag.search.reranker import CrossEncoderReranker
from src.python.insight_rag.graph.knowledge_graph import KnowledgeGraph
from src.python.insight_rag.agents.query_router import QueryRouter

class EnterpriseRAGAgent:
    """Orchestrates ingestion, routing, search, graph traversal, and response assembly."""
    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        reranker: CrossEncoderReranker,
        knowledge_graph: KnowledgeGraph,
        query_router: QueryRouter
    ):
        self.searcher = hybrid_searcher
        self.reranker = reranker
        self.graph = knowledge_graph
        self.router = query_router
        self.memory: List[Dict[str, str]] = []

    def clear_memory(self):
        self.memory = []

    def answer_query(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Processes query intent and compiles answer with citation details."""
        # 1. Intent routing
        routing = self.router.route(query)
        strategy = routing["strategy"]
        
        # 2. Hybrid Retrieval
        candidates = self.searcher.search(query, top_k=top_k * 2)
        candidate_docs = [doc for doc, _ in candidates]
        
        # 3. Graph Traversal Context if requested
        graph_context = ""
        if strategy == "graphrag" and len(self.graph.graph) > 0:
            # Extract query keywords as seeds
            seeds = [w for w in query.split() if len(w) > 4]
            graph_context = self.graph.get_subgraph_context(seeds, max_hops=1)
            
        # 4. Rerank
        reranked = self.reranker.rerank(query, candidate_docs)[:top_k]
        
        # 5. Format Context
        context_blocks = []
        citations = []
        for doc, score in reranked:
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            block = f"[{src} (Page {page})]: {doc.text}"
            context_blocks.append(block)
            
            citations.append({
                "source": src,
                "page": page,
                "score": score
            })
            
        context = "\n\n".join(context_blocks)
        if graph_context:
            context += "\n\nKnowledge Graph Relationships:\n" + graph_context
            
        # 6. Generate extractive synthesis
        answer = f"**Retrieval Strategy:** `{strategy.upper()}`\n"
        answer += f"**Routing Logic:** {routing['reason']}\n\n"
        
        # Pull matching lines
        keywords = [kw.lower() for kw in query.split() if len(kw) > 3]
        found_matches = False
        
        if reranked:
            answer += "Based on the enterprise document indexes, we have synthesized the following:\n\n"
            for doc, _ in reranked:
                sentences = doc.text.split(".")
                for s in sentences:
                    s = s.strip()
                    if any(kw in s.lower() for kw in keywords):
                        answer += f"- {s} (Ref: {doc.metadata.get('source')})\n"
                        found_matches = True
                        break
                if found_matches:
                    break
                    
            if not found_matches:
                # Fallback to snippet
                top_doc = reranked[0][0]
                answer += f"- The primary documentation notes: '{top_doc.text[:200]}...' (Ref: {top_doc.metadata.get('source')})\n"
                
            if graph_context:
                answer += f"\n**Structural Graph Insights:**\n{graph_context}\n"
        else:
            answer += "No matching documents are indexed. Please upload documentation to get specific answers."
            
        # Update memory
        self.memory.append({"query": query, "answer": answer})
        
        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "citations": citations,
            "context": context
        }

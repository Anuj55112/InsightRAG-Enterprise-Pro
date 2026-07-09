import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Package imports
from src.python.insight_rag.config import load_config
from src.python.insight_rag.ingestion.document_loader import DocumentLoader, Document
from src.python.insight_rag.ingestion.chunker import RecursiveChunker
from src.python.insight_rag.search.embedder import SentenceEmbedder
from src.python.insight_rag.search.hybrid_search import HybridSearcher
from src.python.insight_rag.search.reranker import CrossEncoderReranker
from src.python.insight_rag.graph.entity_extractor import EntityRelationExtractor
from src.python.insight_rag.graph.knowledge_graph import KnowledgeGraph
from src.python.insight_rag.agents.query_router import QueryRouter
from src.python.insight_rag.agents.rag_agent import EnterpriseRAGAgent
from src.python.insight_rag.evaluation.evaluator import RAGBenchEvaluator

app = FastAPI(
    title="InsightRAG Pro - Enterprise Retrieval-Augmented Generation API",
    description="REST API for hybrid BM25 + dense search, GraphRAG, entity-relation extraction, and evaluation checks.",
    version="1.0.0"
)

# Load configuration
config = load_config()

# Global instantiations
LOADER = DocumentLoader()
CHUNKER = RecursiveChunker(chunk_size=config.rag_chunk_size, chunk_overlap=config.rag_chunk_overlap)
EMBEDDER = SentenceEmbedder(model_name=config.rag_embedding_model)
SEARCHER = HybridSearcher(EMBEDDER, rrf_k=config.rag_rrf_k)
RERANKER = CrossEncoderReranker(model_name=config.rag_reranker_model)
EXTRACTOR = EntityRelationExtractor()
KNOWLEDGE_GRAPH = KnowledgeGraph()
ROUTER = QueryRouter()

RAG_AGENT = EnterpriseRAGAgent(SEARCHER, RERANKER, KNOWLEDGE_GRAPH, ROUTER)
BENCH_EVALUATOR = RAGBenchEvaluator(RAG_AGENT)

# Document catalog cache
INGESTED_CHUNKS: List[Document] = []

class QueryRequest(BaseModel):
    query: str

class EvalItem(BaseModel):
    query: str
    reference: str

class EvalRequest(BaseModel):
    dataset: List[EvalItem]

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "indexed_chunks": len(INGESTED_CHUNKS),
        "graph_node_count": len(KNOWLEDGE_GRAPH.graph.nodes),
        "graph_edge_count": len(KNOWLEDGE_GRAPH.graph.edges)
    }

@app.post("/ingest")
def ingest_file(file: UploadFile = File(...)):
    """Uploads, parses, chunks, and indexes a document for hybrid retrieval and GraphRAG."""
    # Write temp file
    temp_dir = "data/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
            
        # 1. Load document
        docs = LOADER.load(temp_path)
        if not docs:
            raise HTTPException(status_code=400, detail="Failed to load content from document.")
            
        # 2. Split chunks
        chunks = CHUNKER.split_documents(docs)
        
        # 3. Add to search index
        global INGESTED_CHUNKS
        INGESTED_CHUNKS.extend(chunks)
        SEARCHER.fit(INGESTED_CHUNKS)
        
        # 4. Extract entities and populate GraphRAG
        for chunk in chunks:
            entities, relations = EXTRACTOR.extract(chunk.text)
            for ent in entities:
                KNOWLEDGE_GRAPH.add_entity(ent["name"], ent["type"])
            for ent_a, rel, ent_b in relations:
                KNOWLEDGE_GRAPH.add_relationship(ent_a, rel, ent_b)
                
        # Clean up temp
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "filename": file.filename,
            "status": "ingested",
            "extracted_chunks": len(chunks),
            "total_indexed_chunks": len(INGESTED_CHUNKS),
            "graph_nodes": len(KNOWLEDGE_GRAPH.graph.nodes)
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {e}")

@app.post("/query")
def run_query(request: QueryRequest):
    """Executes agentic routing, hybrid retrieval, cross-encoder ranking, and answer synthesis."""
    try:
        res = RAG_AGENT.answer_query(request.query)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {e}")

@app.get("/graph")
def get_graph():
    """Returns knowledge graph entities and links in JSON format."""
    return KNOWLEDGE_GRAPH.export_graph_json()

@app.post("/evaluate")
def evaluate_rag(request: EvalRequest):
    """Runs evaluation benchmark checks (faithfulness, correctness) over reference QA pairs."""
    try:
        # Convert Request to Dict list
        test_set = [{"query": item.query, "reference": item.reference} for item in request.dataset]
        res = BENCH_EVALUATOR.evaluate_dataset(test_set)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

@app.get("/history")
def get_history():
    """Retrieves current session conversation log memory."""
    return RAG_AGENT.memory

import os
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # RAG parameters
        rag = config_dict.get("rag", {})
        self.rag_chunk_size: int = rag.get("chunk_size", 600)
        self.rag_chunk_overlap: int = rag.get("chunk_overlap", 60)
        self.rag_embedding_model: str = rag.get("embedding_model", "all-MiniLM-L6-v2")
        self.rag_reranker_model: str = rag.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.rag_rrf_k: int = rag.get("rrf_k", 60)
        self.rag_dense_weight: float = rag.get("dense_weight", 0.7)
        self.rag_sparse_weight: float = rag.get("sparse_weight", 0.3)
        
        # Graph parameters
        graph = config_dict.get("graph", {})
        self.graph_pagerank_alpha: float = graph.get("pagerank_alpha", 0.85)
        self.graph_min_community: int = graph.get("min_community_size", 2)
        
        # Evaluation parameters
        eval_dict = config_dict.get("evaluation", {})
        self.eval_relevance_threshold: float = eval_dict.get("relevance_threshold", 0.6)
        self.eval_faithfulness_threshold: float = eval_dict.get("faithfulness_threshold", 0.6)
        
        # API parameter definitions
        api = config_dict.get("api", {})
        self.api_host: str = api.get("host", "0.0.0.0")
        self.api_port: int = api.get("port", 8004)
        
        # UI parameter definitions
        ui = config_dict.get("ui", {})
        self.ui_port: int = ui.get("port", 8505)

def load_config(config_path: str = "") -> Config:
    if not config_path:
        current_dir = Path(__file__).resolve().parent
        config_path = str(current_dir.parent.parent / "configs" / "config.yaml")
        
    if not os.path.exists(config_path):
        return Config({})
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    return Config(config_dict)

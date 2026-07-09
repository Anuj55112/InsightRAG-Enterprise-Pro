import pytest
from src.python.insight_rag.graph.entity_extractor import EntityRelationExtractor
from src.python.insight_rag.graph.knowledge_graph import KnowledgeGraph

def test_entity_extractor():
    text = "Vaswani introduced the self-attention transformer model in the arXiv publication."
    extractor = EntityRelationExtractor()
    entities, relations = extractor.extract(text)
    
    # Check that at least some entities are matched
    entity_names = [e["name"] for e in entities]
    assert "Transformer" in entity_names or "Self-Attention" in entity_names or "Arxiv" in entity_names

def test_knowledge_graph():
    kg = KnowledgeGraph()
    kg.add_entity("Transformer", "Concept")
    kg.add_entity("Vaswani", "Person")
    kg.add_relationship("Transformer", "authored_by", "Vaswani")
    
    context = kg.get_subgraph_context(["Transformer"], max_hops=1)
    assert "Transformer" in context
    assert "Vaswani" in context

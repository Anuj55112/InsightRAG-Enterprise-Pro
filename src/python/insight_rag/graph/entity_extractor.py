import re
from typing import List, Dict, Tuple, Any

class EntityRelationExtractor:
    """Extracts entities and relationships from text using heuristic regex matching."""
    
    def __init__(self):
        # Broad patterns matching ML concepts, methods, entities
        self.entity_patterns = {
            "Concept": r"\b(transformer|self-attention|large language model|llm|neural network|embeddings|attention mechanism|rag|graphrag)\b",
            "Method": r"\b(pagerank|community detection|hybrid search|cross-encoder|reciprocal rank fusion|rrf|bm25|faiss|chromadb)\b",
            "Dataset": r"\b(arxiv|ms-marco|wikipedia|squad|glue|superglue)\b",
            "Person": r"\b(vaswani|dosovitskiy|elovic|zhao|ouyang|brown)\b"
        }

    def extract(self, text: str) -> Tuple[List[Dict[str, str]], List[Tuple[str, str, str]]]:
        """Returns lists of extracted entities and relationship tuples (entity_a, relationship, entity_b)."""
        entities: Dict[str, Dict[str, str]] = {}
        relationships: List[Tuple[str, str, str]] = []
        
        # 1. Extract Entities
        for category, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.lower().strip()
                if entity_name not in entities:
                    entities[entity_name] = {
                        "name": entity_name.title(),
                        "type": category
                    }
                    
        # 2. Extract Relationships (Heuristic sentence-level association)
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            found_entities = [ent for ent in entities if ent in sentence.lower()]
            if len(found_entities) >= 2:
                # Add relationship edges between pairs in sentence
                for i in range(len(found_entities) - 1):
                    ent_a = entities[found_entities[i]]["name"]
                    ent_b = entities[found_entities[i+1]]["name"]
                    
                    # Heuristically classify relation verb
                    rel = "associated_with"
                    if "use" in sentence.lower() or "utilize" in sentence.lower():
                        rel = "uses"
                    elif "implement" in sentence.lower() or "incorporate" in sentence.lower():
                        rel = "implements"
                    elif "improve" in sentence.lower() or "enhance" in sentence.lower():
                        rel = "improves"
                    elif "cites" in sentence.lower() or "reference" in sentence.lower():
                        rel = "references"
                        
                    relationships.append((ent_a, rel, ent_b))
                    
        return list(entities.values()), list(set(relationships))

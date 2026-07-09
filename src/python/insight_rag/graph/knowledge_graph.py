import networkx as nx
from typing import List, Dict, Tuple, Any

class KnowledgeGraph:
    """Builds and analyzes knowledge graphs of entities and semantic relationships using NetworkX."""
    def __init__(self):
        self.graph = nx.Graph()

    def add_entity(self, name: str, entity_type: str):
        """Adds a node to the graph representing an entity."""
        self.graph.add_node(name, type=entity_type)

    def add_relationship(self, ent_a: str, rel: str, ent_b: str):
        """Adds an edge to the graph representing a semantic relationship."""
        # Ensure nodes exist
        if not self.graph.has_node(ent_a):
            self.graph.add_node(ent_a, type="Concept")
        if not self.graph.has_node(ent_b):
            self.graph.add_node(ent_b, type="Concept")
            
        self.graph.add_edge(ent_a, ent_b, relation=rel)

    def get_subgraph_context(self, seed_entities: List[str], max_hops: int = 1) -> str:
        """Retrieves text context from the subgraph surrounding the seed entities."""
        if not seed_entities or len(self.graph) == 0:
            return ""
            
        selected_nodes = set()
        for seed in seed_entities:
            # Match case insensitive
            matched_seed = None
            for node in self.graph.nodes():
                if node.lower() == seed.lower():
                    matched_seed = node
                    break
                    
            if matched_seed:
                selected_nodes.add(matched_seed)
                # Multi-hop retrieval
                sub_nodes = nx.single_source_shortest_path_length(self.graph, matched_seed, cutoff=max_hops)
                selected_nodes.update(sub_nodes.keys())
                
        if not selected_nodes:
            return ""
            
        sub = self.graph.subgraph(selected_nodes)
        
        # Format relationship summary
        summary = []
        for u, v, attrs in sub.edges(data=True):
            summary.append(f"- {u} is {attrs.get('relation', 'connected_to')} {v}")
            
        return "\n".join(summary)

    def detect_communities(self) -> List[List[str]]:
        """Identifies clusters of entities using greedy community modularity."""
        if len(self.graph) < 2:
            return [list(self.graph.nodes())]
            
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(self.graph))
            return [list(c) for c in communities]
        except Exception:
            return [list(self.graph.nodes())]

    def export_graph_json(self) -> Dict[str, Any]:
        """Formats graph structure for Plotly dashboards."""
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "type": attrs.get("type", "Concept")
            })
            
        edges = []
        for u, v, attrs in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "relation": attrs.get("relation", "connected_to")
            })
            
        return {
            "nodes": nodes,
            "edges": edges
        }

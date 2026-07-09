import numpy as np
from typing import List, Union

class SentenceEmbedder:
    """Wrapper around sentence-transformers to encode text chunks."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dim = 384
        
        # Lazy load model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dim = self.model.get_sentence_embedding_dimension() or 384
            print(f"Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            print(f"Could not load SentenceTransformer: {e}. Using numpy random fallback.")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encodes texts into dense embeddings."""
        if isinstance(texts, str):
            texts = [texts]
            
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts)
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                print(f"Embedding encoding failed: {e}. Falling back to random vectors.")
                
        # Generate random mock embeddings (L2 normalized)
        np.random.seed(42)
        random_vecs = np.random.normal(0, 1.0, (len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(random_vecs, axis=1, keepdims=True)
        return random_vecs / (norms + 1e-9)

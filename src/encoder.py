from sentence_transformers import SentenceTransformer
import numpy as np

class VectorEncoder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: list[str]) -> np.ndarray:
        """Converts a list of strings into a float32 numpy array of dense vectors."""
        print(f"Encoding {len(texts)} items... (This may take a moment)")
        # FAISS requires float32 data types
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)
        
    def get_dimension(self) -> int:
        """Returns the vector dimension size (384 for MiniLM)."""
        return self.model.get_sentence_embedding_dimension()
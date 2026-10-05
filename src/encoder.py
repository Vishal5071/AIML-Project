import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class SemanticEncoder:
    """Generates L2-normalized dense embeddings via pretrained Transformer."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Guarantees ||v||_2 = 1.0 for cosine equivalence
        )
        return embeddings.astype(np.float32)
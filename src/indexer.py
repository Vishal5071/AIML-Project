import faiss
import numpy as np

class RecommendationIndices:
    """Manages both Exact Flat and Approximate HNSW FAISS indices."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.exact_index = faiss.IndexFlatIP(dimension)
        self.hnsw_index = None

    def build_exact(self, vectors: np.ndarray):
        """Constructs an exact brute-force inner product index."""
        self.exact_index.reset()
        self.exact_index.add(vectors)

    def build_hnsw(self, vectors: np.ndarray, M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        """Constructs an HNSW graph index."""
        # IndexHNSWFlat with METRIC_INNER_PRODUCT over L2-normalized vectors is exact Cosine
        self.hnsw_index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.hnsw_index.hnsw.efConstruction = ef_construction
        self.hnsw_index.hnsw.efSearch = ef_search
        self.hnsw_index.add(vectors)

    def query_exact(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        return self.exact_index.search(query_vector, top_k)

    def query_hnsw(self, query_vector: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        if self.hnsw_index is None:
            raise RuntimeError("HNSW Index has not been built.")
        return self.hnsw_index.search(query_vector, top_k)
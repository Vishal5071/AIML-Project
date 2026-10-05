import faiss
import numpy as np

class HNSWIndexer:
    def __init__(self, dimension: int, M: int = 32):
        """
        Initializes the HNSW Index.
        :param dimension: Dimensionality of the vectors.
        :param M: Number of bi-directional links created for every new element. 
                  Higher = more accurate but uses more RAM. (32-64 is standard).
        """
        self.dimension = dimension
        
        # Using Inner Product (IP) metric. 
        # When vectors are L2 normalized, Inner Product is equivalent to Cosine Similarity.
        self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200  # Depth of search during index building
        self.index.hnsw.efSearch = 64         # Depth of search during querying
        
    def build_index(self, vectors: np.ndarray):
        """Normalizes vectors and adds them to the HNSW graph."""
        print("Building HNSW Index...")
        # L2 Normalization is REQUIRED for Cosine Similarity via Inner Product
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        print(f"Index built with {self.index.ntotal} items.")
        
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> tuple:
        """Searches the index for the nearest neighbors."""
        # Ensure query is float32 and 2D
        query_vector = np.array([query_vector], dtype=np.float32)
        
        # Normalize the query vector
        faiss.normalize_L2(query_vector)
        
        # Search returns distances (scores) and FAISS index IDs
        distances, indices = self.index.search(query_vector, k=top_k)
        
        return distances[0], indices[0]
import os
import faiss
import numpy as np
import pandas as pd
from src.data_loader import MovieDataLoader
from src.encoder import SemanticEncoder
from src.indexer import RecommendationIndices

class ContentHNSWRecommender:
    """Unified inference service with artifact caching."""
    def __init__(self, data_path: str, artifacts_dir: str = "data/artifacts"):
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        self.df = None
        self.vectors = None
        self.encoder = SemanticEncoder()
        self.indices = RecommendationIndices(self.encoder.embedding_dim)
        self._initialize()

    def _initialize(self):
        meta_path = os.path.join(self.artifacts_dir, "movies_metadata.parquet")
        vec_path = os.path.join(self.artifacts_dir, "embeddings.npy")
        hnsw_path = os.path.join(self.artifacts_dir, "hnsw_index.faiss")

        if os.path.exists(meta_path) and os.path.exists(vec_path) and os.path.exists(hnsw_path):
            print("[INFO] Loading cached embeddings and FAISS index...")
            self.df = pd.read_parquet(meta_path)
            self.vectors = np.load(vec_path)
            self.indices.hnsw_index = faiss.read_index(hnsw_path)
            self.indices.build_exact(self.vectors)
        else:
            print("[INFO] Building artifacts from scratch...")
            loader = MovieDataLoader(self.data_path)
            self.df = loader.load_and_preprocess()
            self.vectors = self.encoder.encode(self.df['dense_document'].tolist())
            self.indices.build_exact(self.vectors)
            self.indices.build_hnsw(self.vectors, M=32, ef_construction=200, ef_search=64)

            # Persist artifacts
            self.df.to_parquet(meta_path)
            np.save(vec_path, self.vectors)
            faiss.write_index(self.indices.hnsw_index, hnsw_path)

        self.title_to_idx = {title.strip().lower(): idx for idx, title in enumerate(self.df['title'])}

    def recommend_by_title(self, title: str, top_k: int = 10, method: str = 'hnsw') -> pd.DataFrame:
        clean_title = title.strip().lower()
        if clean_title not in self.title_to_idx:
            raise KeyError(f"Movie '{title}' not found in database.")

        idx = self.title_to_idx[clean_title]
        query_vec = self.vectors[idx:idx+1]
        search_k = top_k + 1

        if method == 'hnsw':
            scores, indices = self.indices.query_hnsw(query_vec, top_k=search_k)
        elif method == 'exact':
            scores, indices = self.indices.query_exact(query_vec, top_k=search_k)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'hnsw' or 'exact'.")

        result_indices = [i for i in indices[0] if i != idx][:top_k]
        result_scores = [s for i, s in zip(indices[0], scores[0]) if i != idx][:top_k]

        recs = self.df.iloc[result_indices].copy()
        recs['similarity_score'] = result_scores
        return recs[['movieId', 'title', 'genres', 'similarity_score']]
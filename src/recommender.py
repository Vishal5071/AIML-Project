import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.encoder import VectorEncoder
from src.indexer import HNSWIndexer

class ContentBasedRecommender:
    def __init__(self, data_path: str):
        self.df = DataLoader(data_path).load_data()
        self.encoder = VectorEncoder()
        
        # Extract features and build index
        self.vectors = self.encoder.encode(self.df['content'].tolist())
        self.indexer = HNSWIndexer(dimension=self.encoder.get_dimension())
        self.indexer.build_index(self.vectors)
        
        # Create a mapping for quick title lookups
        # Converts to lower case for case-insensitive searching
        self.title_to_idx = {
            title.lower(): idx for idx, title in enumerate(self.df['title'])
        }

    def recommend(self, movie_title: str, top_k: int = 5) -> pd.DataFrame:
        """Returns the top K similar movies based on content."""
        movie_title_lower = movie_title.lower()
        
        if movie_title_lower not in self.title_to_idx:
            # If movie isn't in database, we can encode the raw query directly!
            # This is a major advantage of using semantic embeddings.
            print(f"'{movie_title}' not found. Searching by semantic meaning instead...")
            query_vector = self.encoder.encode([movie_title])[0]
        else:
            # If it is in the database, fetch its pre-calculated vector
            target_idx = self.title_to_idx[movie_title_lower]
            query_vector = self.vectors[target_idx]
            # Add 1 to top_k because the first result will be the movie itself
            top_k += 1  

        # Query the HNSW index
        distances, indices = self.indexer.search(query_vector, top_k=top_k)
        
        # Format the results
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1: continue # FAISS returns -1 if it can't find enough neighbors
            
            movie = self.df.iloc[idx]
            
            # Skip the exact same movie being recommended to the user
            if movie['title'].lower() == movie_title_lower:
                continue
                
            results.append({
                'Movie ID': movie['movieId'],
                'Title': movie['title'],
                'Genres': movie['genres'],
                'Similarity Score': round(float(dist), 4)
            })
            
        return pd.DataFrame(results).head(top_k if movie_title_lower not in self.title_to_idx else top_k-1)
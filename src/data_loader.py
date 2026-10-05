import os
import pandas as pd

class MovieDataLoader:
    """Loads and formats MovieLens data for dense text embeddings."""
    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")
        self.filepath = filepath

    def load_and_preprocess(self) -> pd.DataFrame:
        df = pd.read_csv(self.filepath)
        required_cols = {'movieId', 'title', 'genres'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        df = df.dropna(subset=['title', 'genres']).copy()
        df['genres_cleaned'] = df['genres'].str.replace('|', ' ', regex=False)
        df['genres_list'] = df['genres'].apply(lambda x: [g for g in x.split('|') if g != '(no genres listed)'])

        # Synthesize rich semantic context string: Title + Extracted Genres
        df['dense_document'] = (
            "Movie Title: " + df['title'] + ". Genres: " + df['genres_cleaned'] + "."
        )
        df = df.reset_index(drop=True)
        df['vector_id'] = df.index.astype('int64')
        return df
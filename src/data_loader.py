import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """Loads movies data and creates a combined content feature."""
        print("Loading data...")
        df = pd.read_csv(self.file_path)
        
        # Drop movies with missing titles or genres
        df = df.dropna(subset=['title', 'genres'])
        
        # Replace the MovieLens '|' genre separator with spaces
        df['clean_genres'] = df['genres'].str.replace('|', ' ', regex=False)
        
        # Create the unified content string for embedding
        # Example: "Toy Story (1995) Animation Children's Comedy"
        df['content'] = df['title'] + " " + df['clean_genres']
        
        # Reset index so dataframe index strictly aligns with FAISS integer IDs (0 to N-1)
        df = df.reset_index(drop=True)
        print(f"Loaded {len(df)} movies.")
        return df
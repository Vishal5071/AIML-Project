from src.recommender import ContentBasedRecommender

def main():
    # 1. Provide the path to your MovieLens movies.csv
    # Download from: https://grouplens.org/datasets/movielens/
    data_path = "data/movies.csv"
    
    print("--- Initializing Recommender System ---")
    recommender = ContentBasedRecommender(data_path)
    print("\nSystem ready!\n")
    
    # 2. Define test queries
    test_queries = [
        "Toy Story (1995)",
        "Matrix, The (1999)",
        "Dark Knight, The (2008)",
        # Testing semantic search for a movie not in the DB
        "A movie about spaceships and aliens" 
    ]
    
    # 3. Generate and display recommendations
    for query in test_queries:
        print(f"{'='*50}")
        print(f"Recommendations for: {query}")
        print(f"{'='*50}")
        
        results = recommender.recommend(query, top_k=5)
        print(results.to_string(index=False))
        print("\n")

if __name__ == "__main__":
    main()
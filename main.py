import os
from src.recommender import ContentHNSWRecommender
from evaluation.benchmark import run_comprehensive_benchmark

if __name__ == "__main__":
    dataset_file = "data/movies.csv"

    # Auto-generate a miniature MovieLens sample if the file is missing
    if not os.path.exists(dataset_file):
        print(f"[WARN] {dataset_file} not found. Synthesizing a sample dataset...")
        os.makedirs("data", exist_ok=True)
        sample_df = [
            "movieId,title,genres",
            "1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy",
            "2,Jumanji (1995),Adventure|Children|Fantasy",
            "3,Grumpier Old Men (1995),Comedy|Romance",
            "4,Waiting to Exhale (1995),Comedy|Drama|Romance",
            "5,Father of the Bride Part II (1995),Comedy",
            "6,Heat (1995),Action|Crime|Thriller",
            "7,Sabrina (1995),Comedy|Romance",
            "8,Tom and Huck (1995),Adventure|Children",
            "9,Sudden Death (1995),Action",
            "10,GoldenEye (1995),Action|Adventure|Thriller"
        ]
        with open(dataset_file, "w") as f:
            f.write("\n".join(sample_df))

    # Initialize Engine
    recommender = ContentHNSWRecommender(data_path=dataset_file)

    # Execute Sample Recommendation
    test_movie = recommender.df.iloc[0]['title']
    print(f"\n[QUERY] Fetching Top-5 recommendations for: '{test_movie}'")
    recs = recommender.recommend_by_title(test_movie, top_k=5, method='hnsw')
    print(recs.to_string(index=False))

    # Execute Offline Benchmark
    run_comprehensive_benchmark(recommender, num_queries=min(50, len(recommender.df)), k=5)
import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from src.recommender import ContentHNSWRecommender
from evaluation.metrics import calculate_recall_at_k, calculate_ndcg_at_k

def run_comprehensive_benchmark(recommender: ContentHNSWRecommender, num_queries: int = 500, k: int = 10):
    print(f"\n[BENCHMARK] Executing offline evaluation over {num_queries} random queries (K={k})...")
    np.random.seed(42)
    sample_indices = np.random.choice(len(recommender.vectors), size=num_queries, replace=False)
    queries = recommender.vectors[sample_indices]

    # 1. Evaluate Exact Flat
    start_exact = time.perf_counter()
    exact_scores, exact_ids = recommender.indices.query_exact(queries, top_k=k)
    exact_duration = (time.perf_counter() - start_exact) * 1000  # ms
    avg_exact_latency = exact_duration / num_queries

    # 2. Parameter Sweep on HNSW (efSearch)
    ef_search_values = [8, 16, 32, 64, 128, 256]
    benchmark_results = []

    for ef in ef_search_values:
        recommender.indices.hnsw_index.hnsw.efSearch = ef
        
        start_hnsw = time.perf_counter()
        hnsw_scores, hnsw_ids = recommender.indices.query_hnsw(queries, top_k=k)
        hnsw_duration = (time.perf_counter() - start_hnsw) * 1000 # ms
        avg_hnsw_latency = hnsw_duration / num_queries
        
        recall = calculate_recall_at_k(exact_ids, hnsw_ids)
        ndcg = calculate_ndcg_at_k(exact_ids, exact_scores, hnsw_ids)
        speedup = avg_exact_latency / avg_hnsw_latency
        qps = 1000.0 / avg_hnsw_latency

        benchmark_results.append({
            'efSearch': ef,
            'Recall@10': round(recall, 4),
            'NDCG@10': round(ndcg, 4),
            'Latency (ms)': round(avg_hnsw_latency, 3),
            'QPS': round(qps, 1),
            'Speedup': round(speedup, 2)
        })

    summary_df = pd.DataFrame(benchmark_results)
    print("\n" + tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))
    return summary_df, avg_exact_latency
import numpy as np

def calculate_recall_at_k(exact_indices: np.ndarray, approx_indices: np.ndarray) -> float:
    """Calculates average Recall@K across multiple queries."""
    recalls = []
    for exact, approx in zip(exact_indices, approx_indices):
        intersection = np.intersect1d(exact, approx)
        recalls.append(len(intersection) / len(exact))
    return float(np.mean(recalls))

def calculate_ndcg_at_k(exact_indices: np.ndarray, exact_scores: np.ndarray, approx_indices: np.ndarray) -> float:
    """Calculates NDCG@K using exact similarity scores as ground-truth relevance."""
    ndcgs = []
    for e_idx, e_score, a_idx in zip(exact_indices, exact_scores, approx_indices):
        k = len(e_idx)
        # Map item id -> true relevance
        rel_map = dict(zip(e_idx, e_score))
        
        # Calculate DCG for approximation
        dcg = 0.0
        for rank, item in enumerate(a_idx):
            rel = rel_map.get(item, 0.0)
            dcg += (2**rel - 1) / np.log2(rank + 2)
            
        # Calculate IDCG for ideal ordering
        idcg = 0.0
        for rank, rel in enumerate(e_score):
            idcg += (2**rel - 1) / np.log2(rank + 2)
            
        ndcgs.append(dcg / idcg if idcg > 0 else 1.0)
    return float(np.mean(ndcgs))

def calculate_ils(item_vectors: np.ndarray) -> float:
    """Calculates Intra-List Similarity across recommended items."""
    k = len(item_vectors)
    if k <= 1:
        return 1.0
    sim_matrix = np.dot(item_vectors, item_vectors.T)
    # Sum upper triangular elements excluding diagonal
    upper_sum = np.sum(np.triu(sim_matrix, k=1))
    total_pairs = (k * (k - 1)) / 2
    return float(upper_sum / total_pairs)
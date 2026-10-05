import pytest
import numpy as np
import pandas as pd
import faiss
from src.encoder import SemanticEncoder
from src.indexer import RecommendationIndices

@pytest.fixture(scope="module")
def mock_dataset():
    data = {
        'movieId': [1, 2, 3, 4, 5],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Heat (1995)',
            'Casino (1995)',
            'Lion King, The (1994)'
        ],
        'genres': [
            'Adventure|Animation|Children',
            'Adventure|Children|Fantasy',
            'Action|Crime|Thriller',
            'Crime|Drama',
            'Adventure|Animation|Children'
        ],
        'dense_document': [
            'Movie Title: Toy Story (1995). Genres: Adventure Animation Children.',
            'Movie Title: Jumanji (1995). Genres: Adventure Children Fantasy.',
            'Movie Title: Heat (1995). Genres: Action Crime Thriller.',
            'Movie Title: Casino (1995). Genres: Crime Drama.',
            'Movie Title: Lion King, The (1994). Genres: Adventure Animation Children.'
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def system_components(mock_dataset):
    encoder = SemanticEncoder(model_name='all-MiniLM-L6-v2', device='cpu')
    vectors = encoder.encode(mock_dataset['dense_document'].tolist())
    indices = RecommendationIndices(encoder.embedding_dim)
    indices.build_exact(vectors)
    indices.build_hnsw(vectors, M=16, ef_construction=64, ef_search=32)
    return encoder, vectors, indices

# TEST CASE 1: Mathematical L2 Normalization Verification
def test_l2_normalization(system_components):
    _, vectors, _ = system_components
    norms = np.linalg.norm(vectors, axis=1)
    # Norm of every vector must equal 1.0 within float32 epsilon
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5,
                               err_msg="Vectors are not strictly L2 normalized.")

# TEST CASE 2: Dimensionality Invariant
def test_embedding_dimensions(system_components):
    encoder, vectors, indices = system_components
    assert vectors.shape[1] == 384, "Embedding dimensionality must be 384 for all-MiniLM-L6-v2"
    assert indices.dimension == 384, "Index dimension mismatch"

# TEST CASE 3: Identity Retrieval (Self-Query Return Score ~ 1.0)
def test_self_identity_retrieval(system_components):
    _, vectors, indices = system_components
    for i in range(len(vectors)):
        query = vectors[i:i+1]
        scores, ids = indices.query_exact(query, top_k=1)
        assert ids[0][0] == i, f"Expected self-id {i}, got {ids[0][0]}"
        assert np.isclose(scores[0][0], 1.0, atol=1e-4), f"Self similarity must be 1.0, got {scores[0][0]}"

# TEST CASE 4: Semantic Clustering Integrity
def test_semantic_genre_affinity(system_components):
    """Toy Story (idx 0) should be semantically closer to Lion King (idx 4) than to Casino (idx 3)."""
    _, vectors, indices = system_components
    query = vectors[0:1] # Toy Story
    scores, ids = indices.query_exact(query, top_k=5)
    result_ids = ids[0].tolist()
    
    assert result_ids.index(4) < result_ids.index(3), \
        "Animation child movie (Lion King) should rank higher than Crime Drama (Casino) for Toy Story query"

# TEST CASE 5: Boundary & Exception Handling
def test_index_out_of_bounds_and_empty(system_components):
    _, _, indices = system_components
    dummy_query = np.zeros((1, 384), dtype=np.float32)
    scores, ids = indices.query_hnsw(dummy_query, top_k=100) # Request K > N
    assert len(scores[0]) == 100
    # FAISS pads with -1 when k exceeds index population
    assert -1 in ids[0]

# TEST CASE 6: Empirical Accuracy Verification (Recall@K Sanity Check)
def test_hnsw_exact_recall_threshold(system_components):
    _, vectors, indices = system_components
    query = vectors[0:1]
    _, exact_ids = indices.query_exact(query, top_k=3)
    _, hnsw_ids = indices.query_hnsw(query, top_k=3)
    
    intersection = set(exact_ids[0]).intersection(set(hnsw_ids[0]))
    recall = len(intersection) / 3.0
    assert recall >= 0.66, f"HNSW failed minimum recall threshold: {recall}"
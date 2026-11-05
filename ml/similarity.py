import numpy as np

def cosine_similarity_scores(query_embedding, embeddings):
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D movie embeddings, got {embeddings.shape}")

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"query={query_embedding.shape}, "
            f"movies={embeddings.shape}"
        )

    return embeddings @ query_embedding


def rank_indices(scores, top_k=5, exclude_index=None):
    scores = np.asarray(scores).copy()

    if exclude_index is not None:
        scores[exclude_index] = -np.inf

    return np.argsort(scores)[::-1][:top_k]
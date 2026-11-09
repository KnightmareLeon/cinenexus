"""Compare SBERT vs TF-IDF semantic search on sample queries.

Runs a side-by-side baseline evaluation: for each sample query, print the
top-k movies ranked by SBERT cosine similarity and by TF-IDF cosine similarity,
plus ranking correlation and per-query inference time.

SBERT embeddings are loaded from the cached data/embeddings.npz file instead
of re-encoding all movie texts, avoiding high memory usage.

This script expects the SQLite database (cinenexus.db) and embeddings.npz to
already exist in data/, populated by scripts/create_database.py,
scripts/import_dataset.py, and streamlit run app.py respectively.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repositories.sqlite_movie_repository import SQLiteMovieRepository
from repositories.embedding_repository import EmbeddingRepository
from services.movie_service import MovieService
from ml.model import load_model, MODEL_NAME
from ml.similarity import cosine_similarity_scores, rank_indices
from ml.tfidf import build_tfidf_matrix, compute_tfidf_scores

SAMPLE_QUERIES = [
    "space adventure about family",
    "heist movie with clever planning",
    "romantic comedy with funny dialogue",
    "post-apocalyptic survival story",
    "biopic about a scientist",
]

TOP_K = 5


def spearman_correlation(rank_a, rank_b):
    """Compute Spearman rank correlation between two ranked lists of movie IDs."""
    a_pos = {movie_id: i for i, movie_id in enumerate(rank_a)}
    b_pos = {movie_id: i for i, movie_id in enumerate(rank_b)}
    common = [m for m in rank_a if m in b_pos]
    if len(common) < 2:
        return None
    d_sq = sum((a_pos[m] - b_pos[m]) ** 2 for m in common)
    n = len(common)
    return 1 - (6 * d_sq) / (n * (n * n - 1))


def run_baseline_comparison(data_dir):
    db_path = Path(data_dir) / "cinenexus.db"
    embeddings_path = Path(data_dir) / "embeddings.npz"

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        print("Run scripts/create_database.py and scripts/import_dataset.py first.")
        sys.exit(1)

    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found: {embeddings_path}")
        print("Run streamlit run app.py first to generate embeddings.npz.")
        sys.exit(1)

    repository = SQLiteMovieRepository(db_path)
    movie_service = MovieService(repository)

    movie_count = movie_service.get_movie_count()
    print(f"Loaded {movie_count:,} movies from {db_path}")
    print(f"Sample queries: {len(SAMPLE_QUERIES)}")
    print(f"Top-K per query: {TOP_K}")
    print("=" * 70)

    # --- Get search texts from SQLite ---
    search_texts = movie_service.get_search_texts()
    print(f"Retrieved {len(search_texts):,} search texts from database")

    # --- SBERT embeddings (load from cache) ---
    print("[SBERT] Loading cached embeddings...")
    embedding_repo = EmbeddingRepository(embeddings_path)
    metadata = movie_service.get_embedding_metadata()
    fingerprint = metadata.get("fingerprint", "")
    sbert_embeddings = embedding_repo.load(fingerprint)
    if sbert_embeddings is None:
        print("  Embeddings not found or fingerprint mismatch; generating...")
        model = load_model()
        from ml.embeddings import encode_texts
        t0 = time.time()
        sbert_embeddings = encode_texts(search_texts, model)
        sbert_time = time.time() - t0
        embedding_repo.save(sbert_embeddings, fingerprint)
    else:
        sbert_time = 0.0
        model = load_model()  # still needed for query encoding
    print(f"[SBERT] Embeddings ready: {sbert_embeddings.shape} (load time: {sbert_time:.2f}s)")

    # --- TF-IDF matrix ---
    print("[TF-IDF] Building TF-IDF matrix...")
    t0 = time.time()
    tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(
        search_texts, lowercase=True, strip_accents="unicode", stop_words="english"
    )
    tfidf_build_time = time.time() - t0
    print(
        f"[TF-IDF] Matrix ready: {tfidf_matrix.shape} in {tfidf_build_time:.2f}s"
    )

    print("=" * 70)
    print(f"{'Query':<40} {'SBERT Top':<40} {'TF-IDF Top':<40}")
    print("-" * 120)

    total_sbert_query_time = 0.0
    total_tfidf_query_time = 0.0
    correlations = []

    for query in SAMPLE_QUERIES:
        # --- SBERT query ---
        t0 = time.time()
        from ml.embeddings import encode_query
        query_emb = encode_query(query, model)
        sbert_scores = cosine_similarity_scores(query_emb, sbert_embeddings)
        sbert_indices = rank_indices(sbert_scores, TOP_K)
        sbert_query_time = time.time() - t0
        total_sbert_query_time += sbert_query_time

        ranked_ids = movie_service.get_movie_ids_by_embedding_indices(
            [int(i) for i in sbert_indices]
        )
        ranked_movies = movie_service.get_movies_by_ids(ranked_ids)
        sbert_top = [m.title for m in ranked_movies]
        sbert_ids = [m.id for m in ranked_movies]

        # --- TF-IDF query ---
        t0 = time.time()
        tfidf_scores = compute_tfidf_scores(
            tfidf_vectorizer, tfidf_matrix, [query]
        )
        tfidf_indices = rank_indices(tfidf_scores, TOP_K)
        tfidf_query_time = time.time() - t0
        total_tfidf_query_time += tfidf_query_time

        ranked_ids = movie_service.get_movie_ids_by_embedding_indices(
            [int(i) for i in tfidf_indices]
        )
        ranked_movies = movie_service.get_movies_by_ids(ranked_ids)
        tfidf_top = [m.title for m in ranked_movies]
        tfidf_ids = [m.id for m in ranked_movies]

        # --- Correlation ---
        corr = spearman_correlation(sbert_ids, tfidf_ids)
        correlations.append(corr)

        sbert_str = " | ".join(sbert_top[:TOP_K])[:40]
        tfidf_str = " | ".join(tfidf_top[:TOP_K])[:40]
        print(f"{query:<40} {sbert_str:<40} {tfidf_str:<40}")

    print("=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"SBERT embedding load time:  {sbert_time:.2f}s")
    print(f"TF-IDF build time:          {tfidf_build_time:.2f}s")
    print(f"Avg SBERT query time:       {total_sbert_query_time / len(SAMPLE_QUERIES):.4f}s")
    print(f"Avg TF-IDF query time:      {total_tfidf_query_time / len(SAMPLE_QUERIES):.4f}s")
    valid_corrs = [c for c in correlations if c is not None]
    if valid_corrs:
        print(f"Avg Spearman correlation:   {np.mean(valid_corrs):.4f}")
        print(f"Min Spearman correlation:   {np.min(valid_corrs):.4f}")
        print(f"Max Spearman correlation:   {np.max(valid_corrs):.4f}")
    else:
        print("Avg Spearman correlation:   N/A (no overlapping results)")
    print("=" * 70)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    run_baseline_comparison(str(data_dir))
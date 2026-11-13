import re
import numpy as np

from domain.recommendation import MovieMatch
from ml.embeddings import encode_query
from ml.similarity import cosine_similarity_scores, rank_indices

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "he", "her", "his", "in", "is", "it", "its", "of", "on", "or",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "who", "will", "with", "movie", "film", "about", "story", "man",
    "woman", "young", "old", "life", "find", "finds", "gets", "into",
}


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    return tokens - _STOPWORDS


def _catalog_prior(num_items: int) -> np.ndarray:
    """
    Smooth logarithmic prominence curve calibrated for the 950k catalog.
    Mainstream, well-annotated releases sit in the first ~15,000 indices,
    while unannotated stub entries sit in the long tail (100,000+).
    """
    indices = np.arange(num_items, dtype=np.float32)
    decay = 1.0 / (1.0 + np.power(indices / 8000.0, 0.85))
    return 0.14 * decay


def search_movies_tfidf(query, movie_service, index, top_k=5):
    indices, scores = index.search(query, top_k)
    movie_ids = movie_service.get_movie_ids_by_embedding_indices(indices)
    movies = movie_service.get_movies_by_ids(movie_ids)
    scores_by_id = dict(zip(movie_ids, scores))
    return [
        MovieMatch(movie=movie, similarity=float(scores_by_id[movie.id]))
        for movie in movies
    ]


def search_movies(query, movie_service, embeddings, model, top_k=5):
    query_embedding = encode_query(query, model)
    raw_scores = cosine_similarity_scores(query_embedding, embeddings)

    # Stage 1: Retrieve top candidate pool using semantic similarity + calibrated catalog prior
    pool_size = max(top_k * 20, 150)
    stage1_scores = raw_scores + _catalog_prior(len(raw_scores))
    candidate_indices = rank_indices(stage1_scores, pool_size)

    movie_ids = movie_service.get_movie_ids_by_embedding_indices(
        candidate_indices.tolist()
    )
    movies = movie_service.get_movies_by_ids(movie_ids)
    index_by_id = dict(zip(movie_ids, candidate_indices.tolist()))

    # Stage 2: Re-rank candidates using Actor, Director, Genre, Theme & Title overlap
    query_lower = query.strip().casefold()
    query_tokens = _tokenize(query)

    scored_candidates = []
    for movie in movies:
        idx = index_by_id.get(movie.id)
        if idx is None:
            continue

        base_sim = float(raw_scores[idx])
        prior_bonus = float(stage1_scores[idx] - raw_scores[idx])
        meta_bonus = 0.0

        if query_tokens:
            # 1. Title match boost
            title_tokens = _tokenize(movie.title)
            if title_tokens and query_tokens & title_tokens:
                overlap_ratio = len(query_tokens & title_tokens) / max(len(title_tokens), 1)
                meta_bonus += 0.08 * overlap_ratio

            # 2. Director match boost
            if movie.director:
                dir_lower = movie.director.casefold()
                dir_tokens = _tokenize(movie.director)
                if dir_lower in query_lower or (
                    dir_tokens and len(query_tokens & dir_tokens) >= len(dir_tokens)
                ):
                    meta_bonus += 0.12

            # 3. Cast / Actor match boost (top 10 billed actors)
            for actor in (movie.actors or ())[:10]:
                actor_lower = actor.casefold()
                actor_tokens = _tokenize(actor)
                if actor_lower in query_lower or (
                    len(actor_tokens) >= 2 and actor_tokens.issubset(query_tokens)
                ):
                    meta_bonus += 0.11
                    break

            # 4. Genre & Theme match boost
            movie_genres_themes = _tokenize(" ".join(movie.genres or ())) | _tokenize(
                " ".join(movie.themes or ())
            )
            if ("sci" in query_tokens or "scifi" in query_tokens) and "science" in movie_genres_themes:
                meta_bonus += 0.05
            genre_hits = len(query_tokens & movie_genres_themes)
            if genre_hits:
                meta_bonus += min(0.035 * genre_hits, 0.07)

        # Penalize stub records with ultra-short descriptions (< 55 chars)
        overview_len = len((movie.overview or "").strip())
        completeness_adj = -0.08 if overview_len < 55 else 0.015

        final_score = min(base_sim + prior_bonus + meta_bonus + completeness_adj, 0.999)
        scored_candidates.append((final_score, MovieMatch(movie=movie, similarity=final_score)))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [match for _, match in scored_candidates[:top_k]]


def get_similar_movies(movie_id, movie_service, embeddings, top_k=5):
    selected_index = movie_service.get_movie_embedding_index(movie_id)

    if selected_index is None or selected_index < 0 or selected_index >= len(embeddings):
        return []

    source_movie = movie_service.get_movie_by_id(movie_id)
    raw_scores = cosine_similarity_scores(embeddings[selected_index], embeddings)

    # Pull top 120 candidates using semantic similarity + catalog prominence prior
    pool_size = max(top_k * 20, 120)
    stage1_scores = raw_scores + _catalog_prior(len(raw_scores))
    candidate_indices = rank_indices(
        stage1_scores, pool_size, exclude_index=selected_index
    )

    movie_ids = movie_service.get_movie_ids_by_embedding_indices(
        candidate_indices.tolist()
    )
    movies = movie_service.get_movies_by_ids(movie_ids)
    index_by_id = dict(zip(movie_ids, candidate_indices.tolist()))

    if source_movie is None:
        return [
            MovieMatch(movie=movie, similarity=float(raw_scores[index_by_id[movie.id]]))
            for movie in movies[:top_k]
            if movie.id in index_by_id
        ]

    src_genres = set(source_movie.genres or ())
    src_actors = set(source_movie.actors or ())
    src_themes = set(source_movie.themes or ())
    src_director = source_movie.director

    scored_candidates = []
    for movie in movies:
        idx = index_by_id.get(movie.id)
        if idx is None:
            continue

        base_sim = float(raw_scores[idx])
        prior_bonus = float(stage1_scores[idx] - raw_scores[idx]) * 0.65
        graph_bonus = 0.0

        if src_director and movie.director and src_director == movie.director:
            graph_bonus += 0.08

        if src_genres and movie.genres:
            shared_g = len(src_genres & set(movie.genres))
            graph_bonus += min(0.025 * shared_g, 0.06)

        if src_actors and movie.actors:
            shared_a = len(src_actors & set(movie.actors))
            graph_bonus += min(0.04 * shared_a, 0.08)

        if src_themes and movie.themes:
            shared_t = len(src_themes & set(movie.themes))
            graph_bonus += min(0.02 * shared_t, 0.04)

        final_score = min(base_sim + prior_bonus + graph_bonus, 0.999)
        scored_candidates.append((final_score, MovieMatch(movie=movie, similarity=final_score)))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    return [match for _, match in scored_candidates[:top_k]]
from domain.movie import Movie
from domain.recommendation import MovieMatch

from ml.embeddings import encode_query
from ml.similarity import (cosine_similarity_scores, rank_indices)


def search_movies(query: str, movies: list[Movie], embeddings, model, top_k: int = 5) -> list[MovieMatch]:
    query_embedding = encode_query(query, model)

    scores = cosine_similarity_scores(query_embedding, embeddings)

    ranked_indices = rank_indices(scores, top_k)

    return [MovieMatch(movie=movies[index], similarity=float(scores[index])) for index in ranked_indices]

def get_similar_movies(movie_id: str, movies: list[Movie], embeddings, top_k: int = 5) -> list[MovieMatch]:
    selected_idx = next((index for index, movie in enumerate(movies) if movie.id == str(movie_id)), None)

    if selected_idx is None:
        return []

    scores = cosine_similarity_scores([embeddings[selected_idx]], embeddings)

    ranked_indices = rank_indices(scores, top_k, exclude_index=selected_idx)

    return [MovieMatch(movie=movies[index], similarity=float(scores[index]))for index in ranked_indices]
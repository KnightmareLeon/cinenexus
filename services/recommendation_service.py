from domain.recommendation import MovieMatch

from ml.embeddings import encode_query
from ml.similarity import cosine_similarity_scores, rank_indices

def search_movies(query, movie_service, embeddings, model, top_k=5):
    query_embedding = encode_query(query, model)

    scores = cosine_similarity_scores(query_embedding, embeddings)

    ranked_indices = rank_indices(scores, top_k)

    movie_ids = movie_service.get_movie_ids_by_embedding_indices(ranked_indices.tolist())

    movies = movie_service.get_movies_by_ids(movie_ids)

    return [MovieMatch(movie=movie, similarity=float(scores[index])) for movie, index in zip(movies, ranked_indices)]

def get_similar_movies(movie_id, movie_service, embeddings,top_k=5):
    selected_index = movie_service.get_movie_embedding_index(movie_id)

    if selected_index is None:
        return []

    if selected_index < 0 or selected_index >= len(embeddings):
        return []

    scores = cosine_similarity_scores(embeddings[selected_index], embeddings)

    ranked_indices = rank_indices(scores, top_k, exclude_index=selected_index)

    movie_ids = movie_service.get_movie_ids_by_embedding_indices(ranked_indices.tolist())

    movies = movie_service.get_movies_by_ids(movie_ids)

    return [MovieMatch(movie=movie, similarity=float(scores[index])) for movie, index in zip(movies, ranked_indices)]
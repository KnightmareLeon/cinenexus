from ml.embeddings import encode_texts
from repositories.embedding_repository import EmbeddingRepository

def get_movie_embeddings(texts, model, model_name, repository: EmbeddingRepository):
    fingerprint = repository.create_fingerprint(texts, model_name)

    embeddings = repository.load(fingerprint)

    if embeddings is not None:
        return embeddings

    embeddings = encode_texts(texts, model)

    repository.save(embeddings, fingerprint)

    return embeddings

def get_movie_embeddings_from_database(movie_service, model, model_name, repository: EmbeddingRepository, expected_fingerprint,):

    embeddings = repository.load(expected_fingerprint)

    if embeddings is not None:
        return embeddings

    actual_fingerprint = repository.create_fingerprint(
        movie_service.iter_search_texts(), model_name
    )

    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "The database embedding fingerprint does not match "
            "the current movie dataset. Please re-import the dataset."
        )

    texts = movie_service.get_search_texts()
    embeddings = encode_texts(texts, model)

    repository.save(embeddings, actual_fingerprint)

    return embeddings

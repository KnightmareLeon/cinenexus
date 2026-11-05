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
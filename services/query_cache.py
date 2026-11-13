import streamlit as st
from pathlib import Path

from services.graph_service import build_graph
from services.movie_service import MovieService
from services.dataset_export_service import movies_to_csv
from services.recommendation_service import get_similar_movies, search_movies, search_movies_tfidf
from services.embedding_service import get_movie_embeddings_from_database

from repositories.embedding_repository import EmbeddingRepository

from ml.tfidf import TfidfIndex
from ml.model import load_model, MODEL_NAME


@st.cache_resource(show_spinner="Building the TF-IDF index for this dataset...", max_entries=1)
def get_tfidf_index(dataset_fingerprint: str, _movie_service):
    return TfidfIndex(_movie_service.iter_search_texts())

@st.cache_resource(show_spinner=f"Getting {MODEL_NAME} model...")
def get_cached_model():
    return load_model()

@st.cache_resource(show_spinner=f"Retrieving embeddings for {MODEL_NAME}...", max_entries=1)
def load_cached_embeddings(fingerprint, _movie_service, model_name, embeddings_path: Path):
    repository = EmbeddingRepository(embeddings_path)

    return get_movie_embeddings_from_database(
        movie_service=_movie_service,
        model=get_cached_model(),
        model_name=model_name,
        repository=repository,
        expected_fingerprint=fingerprint,
    )

@st.cache_data(show_spinner=False)
def get_cached_tfidf_search(query, top_k, dataset_fingerprint, _movie_service, _index):
    return search_movies_tfidf(query, _movie_service, _index, top_k)

@st.cache_data(show_spinner="Acquiring movie data from saved dataset...")
def get_cached_movies_page(_movie_service: MovieService, page: int, page_size: int, dataset_fingerprint: str):
    return _movie_service.get_movies_page(page=page, page_size=page_size)

@st.cache_data(show_spinner=False)
def get_cached_title_search(_movie_service: MovieService, query: str, limit: int, dataset_fingerprint: str):
    return _movie_service.search_movies_by_title(query=query, limit=limit)

@st.cache_data(show_spinner="Preparing the full movie catalog CSV...", max_entries=1)
def get_cached_catalog_csv(_movie_service: MovieService, dataset_fingerprint: str) -> bytes:
    return movies_to_csv(_movie_service.iter_movies())

@st.cache_data(show_spinner=False)
def get_cached_semantic_search(query: str, top_k: int, dataset_fingerprint: str, _movie_service, _embeddings, _model):
    return search_movies(query, _movie_service, _embeddings, _model, top_k)

@st.cache_data(show_spinner=False)
def get_cached_similar_movies(movie_id: str, top_k: int, dataset_fingerprint: str, _movie_service, _embeddings):
    return get_similar_movies(movie_id, _movie_service, _embeddings, top_k)

@st.cache_data(show_spinner="Building knowledge graph for chosen movie...")
def get_cached_movie_graph(_movie_service: MovieService, _embeddings, movie_id: str,  dataset_fingerprint: str, max_related_movies: int = 20):
    return build_graph(focus_movie_id=movie_id, embeddings=_embeddings, movie_service=_movie_service, max_related_movies = max_related_movies)
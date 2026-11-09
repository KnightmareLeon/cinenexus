import streamlit as st

from services.graph_service import build_graph
from services.movie_service import MovieService
from services.recommendation_service import get_similar_movies, search_movies
from services.recommendation_service import search_movies_tfidf
from ml.tfidf import TfidfIndex


@st.cache_resource(show_spinner="Building the TF-IDF index for this dataset…", max_entries=1)
def get_tfidf_index(dataset_fingerprint: str, _movie_service):
    return TfidfIndex(_movie_service.iter_search_texts())


@st.cache_data(show_spinner=False)
def get_cached_tfidf_search(query, top_k, dataset_fingerprint, _movie_service):
    index = get_tfidf_index(dataset_fingerprint, _movie_service)
    return search_movies_tfidf(query, _movie_service, index, top_k)

@st.cache_data
def get_cached_movies_page(_movie_service: MovieService, page: int, page_size: int, dataset_fingerprint: str,):
    return _movie_service.get_movies_page(page=page, page_size=page_size)

@st.cache_data(show_spinner=False)
def get_cached_title_search(_movie_service: MovieService, query: str, limit: int, dataset_fingerprint: str,):
    return _movie_service.search_movies_by_title(query=query, limit=limit)

@st.cache_data(show_spinner=False)
def get_cached_semantic_search(query: str, top_k: int, dataset_fingerprint: str, _movie_service, _embeddings, _model,):
    return search_movies(query, _movie_service, _embeddings, _model, top_k)

@st.cache_data(show_spinner=False)
def get_cached_similar_movies(movie_id: str, top_k: int, dataset_fingerprint: str, _movie_service, _embeddings,):
    return get_similar_movies(movie_id, _movie_service, _embeddings, top_k)

@st.cache_data
def get_cached_movie_graph(_movie_service: MovieService, movie_id: str, dataset_fingerprint: str,):
    return build_graph(focus_movie_id=movie_id, movie_service=_movie_service)

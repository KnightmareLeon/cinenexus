import streamlit as st

from services.graph_service import build_graph
from services.movie_service import MovieService
from services.recommendation_service import get_similar_movies, search_movies

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
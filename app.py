import streamlit as st

from pathlib import Path

from repositories.sqlite_movie_repository import SQLiteMovieRepository
from repositories.embedding_repository import EmbeddingRepository

from presentation.search_page import render_search_page
from presentation.graph_page import render_graph_page
from presentation.dataset_page import render_dataset_page
from presentation.metrics_page import render_metrics_page

from services.movie_service import MovieService
from services.embedding_service import get_movie_embeddings_from_database

from ml.model import load_model, MODEL_NAME

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "cinenexus.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"

@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_embeddings(fingerprint, _movie_service, model_name):
    repository = EmbeddingRepository(EMBEDDINGS_PATH)

    return get_movie_embeddings_from_database(movie_service=_movie_service, model=get_model(), model_name=model_name, repository=repository, expected_fingerprint=fingerprint)

try:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}. "
            "Run the database creation and dataset import scripts first."
        )

    movie_repository = SQLiteMovieRepository(DATABASE_PATH)
    movie_service = MovieService(movie_repository)

    movie_count = movie_service.get_movie_count()

    if movie_count == 0:
        raise ValueError(
            "The movie database is empty. "
            "Run the dataset importer first."
        )

    embedding_metadata = movie_service.get_embedding_metadata()

    if not embedding_metadata.get("fingerprint"):
        raise ValueError(
            "Embedding metadata is missing. "
            "Please re-import the dataset."
        )

    if embedding_metadata.get("model_name") != MODEL_NAME:
        raise ValueError(
            "The embedding model recorded in the database does not match "
            "the current model. Please re-import the dataset."
        )

    if int(embedding_metadata.get("embedding_count", -1)) != movie_count:
        raise ValueError(
            "The number of movies does not match the embedding metadata. "
            "Please re-import the dataset."
        )

    dataset_fingerprint = embedding_metadata["fingerprint"]

except Exception as exc:
    st.error(f"Could not initialize CineNexus: {exc}")
    st.stop()

try:
    model = get_model()

    embeddings = load_embeddings(embedding_metadata["fingerprint"], movie_service, MODEL_NAME)

    if len(embeddings) != movie_count:
        raise ValueError(
            "Embedding count does not match the number of movies "
            "in the database."
        )

except Exception as exc:
    st.error(f"Could not load movie embeddings: {exc}")
    st.stop()

st.set_page_config(page_title="CineNexus", page_icon="🎬", layout="wide")

st.title("🎬 CineNexus")
st.caption("Explore movies through semantic similarity and connected entities.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Movies loaded:** {movie_service.get_movie_count():,}")
    st.write("**Embedding model:** `all-MiniLM-L6-v2`")
    top_k = st.slider("Similar movies to show", min_value=3, max_value=10, value=5)

tab_search, tab_metrics, tab_graph, tab_dataset = st.tabs(["🔎 Movie Search", "📊 Search Metrics", "🕸️ Knowledge Graph", "📚 Dataset"])

with tab_search:
    render_search_page(movie_service, embeddings, model, top_k, dataset_fingerprint)

with tab_metrics:
    render_metrics_page(movie_service, embeddings, model, top_k, dataset_fingerprint)

with tab_graph:
    render_graph_page(movie_service, dataset_fingerprint)

with tab_dataset:
    render_dataset_page(movie_service, dataset_fingerprint)

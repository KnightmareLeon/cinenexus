
import streamlit as st
from pathlib import Path

from repositories.sqlite_movie_repository import SQLiteMovieRepository

from presentation.search_page import render_search_page
from presentation.graph_page import render_graph_page
from presentation.dataset_page import render_dataset_page
from presentation.metrics_page import render_metrics_page

from services.movie_service import MovieService
from services.query_cache import get_cached_model, load_cached_embeddings, get_tfidf_index
from services.evaluation_service import generate_imdb_test_cases

from ml.model import MODEL_NAME

st.set_page_config(page_title="CineNexus", page_icon=":material/movie:", layout="wide")

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "cinenexus.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"

IMDB_SOURCE_PATH = DATA_DIR / "imdb_movies.csv"
IMDB_TEST_CASES_PATH = DATA_DIR / "imdb_test_cases.csv"
IMDB_SKIPPED_PATH = DATA_DIR / "imdb_skipped_cases.csv"

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


if IMDB_SOURCE_PATH.exists() and not IMDB_TEST_CASES_PATH.exists():
    try:
        with st.spinner("Generating IMDb evaluation test cases..."):
            result = generate_imdb_test_cases(
                IMDB_SOURCE_PATH,
                movie_service,
                output_path=IMDB_TEST_CASES_PATH,
                skipped_path=IMDB_SKIPPED_PATH,
                sample_size=200,
            )

        st.success(f"Generated {len(result['test_cases'])} IMDb test cases.")

    except Exception as exc:
        st.warning(f"Could not generate IMDb test cases: {exc}")

try:
    model = get_cached_model()
    tfidf_index = get_tfidf_index(dataset_fingerprint, movie_service)
    embeddings = load_cached_embeddings(embedding_metadata["fingerprint"], movie_service, MODEL_NAME, EMBEDDINGS_PATH)

    if len(embeddings) != movie_count:
        raise ValueError(
            "Embedding count does not match the number of movies "
            "in the database."
        )

except Exception as exc:
    st.error(f"Could not load movie embeddings: {exc}")
    st.stop()

st.title(":material/movie: CineNexus")
st.caption("Explore movies through semantic similarity and connected entities.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Movies loaded:** {movie_service.get_movie_count():,}")
    st.write("**Embedding model:** `all-MiniLM-L6-v2`")
    top_k = st.slider("Similar movies to show", min_value=3, max_value=10, value=5)

tab_search, tab_metrics, tab_graph, tab_dataset = st.tabs([":material/search: Movie Search", ":material/analytics: Search Metrics", ":material/account_tree: Knowledge Graph", ":material/database: Dataset"])

with tab_search:
    render_search_page(movie_service, embeddings, model, top_k, dataset_fingerprint, tfidf_index)

with tab_metrics:
    render_metrics_page(movie_service, embeddings, model, dataset_fingerprint, tfidf_index)

with tab_graph:
    render_graph_page(movie_service, embeddings, dataset_fingerprint)

with tab_dataset:
    render_dataset_page(movie_service, dataset_fingerprint)
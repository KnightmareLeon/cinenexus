import streamlit as st

from pathlib import Path

from repositories.movie_repository import MovieRepository
from repositories.embedding_repository import EmbeddingRepository

from presentation.search_page import render_search_page
from presentation.graph_page import render_graph_page
from presentation.dataset_page import render_dataset_page

from services.graph_service import build_movie_relationship_index
from services.embedding_service import get_movie_embeddings

from ml.model import load_model, MODEL_NAME

st.set_page_config(page_title="CineNexus", page_icon="🎬", layout="wide")

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_PATH = BASE_DIR / "data" / "embeddings.npz"

@st.cache_data
def load_movie_data(data_dir):
    repository = MovieRepository(Path(data_dir))
    movies_df = repository.load_dataframe()
    movies = repository.get_all(movies_df)
    return movies_df, movies


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def load_embeddings(texts_tuple, model_name):
    repository = EmbeddingRepository(EMBEDDINGS_PATH)

    return get_movie_embeddings(
        texts=texts_tuple,
        model=get_model(),
        model_name=model_name,
        repository=repository
    )

@st.cache_resource
def load_relationship_index(_movies):
    return build_movie_relationship_index(_movies)

try:
    movies_df, movies = load_movie_data(str(DATA_DIR))
except Exception as exc:
    st.error(f"Could not load movies: {exc}")
    st.stop()

model = get_model()

embeddings = load_embeddings(tuple(movies_df["search_text"].tolist()), MODEL_NAME)
relationship_index = load_relationship_index(movies)

st.title("🎬 CineNexus")
st.caption("Explore movies through semantic similarity and connected entities.")

with st.sidebar:
    st.header("Settings")
    st.write(f"**Movies loaded:** {len(movies)}")
    st.write("**Embedding model:** `all-MiniLM-L6-v2`")
    top_k = st.slider("Similar movies to show", min_value=3, max_value=10, value=5)

tab_search, tab_graph, tab_dataset = st.tabs(["🔎 Movie Search", "🕸️ Knowledge Graph", "📚 Dataset"])

with tab_search:
    render_search_page(movies, embeddings, load_model(), top_k)

with tab_graph:
    render_graph_page(movies, relationship_index)

with tab_dataset:
    render_dataset_page(movies_df)
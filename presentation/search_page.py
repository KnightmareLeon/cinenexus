
import streamlit as st

from services.movie_service import MovieService
from services.recommendation_service import (search_movies, get_similar_movies,)
from services.query_cache import (get_cached_semantic_search, get_cached_title_search, get_cached_similar_movies)

@st.fragment
def render_semantic_search(movie_service: MovieService, embeddings, model, top_k: int, dataset_fingerprint: str):
    st.subheader("Find a movie")

    with st.form("semantic_search_form"):
        query = st.text_input("Search by title or describe what you want to watch", placeholder="e.g., a space adventure about family")

        submitted = st.form_submit_button("Search")

    if submitted:
        if query.strip():
            results = get_cached_semantic_search(query.strip(), top_k, dataset_fingerprint, movie_service, embeddings, model)

            st.markdown("**Best semantic matches**")

            if not results:
                st.info("No matching movies found.")
                return

            for match in results:
                movie = match.movie

                st.markdown(
                    f"**{movie.title} ({movie.year})** "
                    f"· similarity `{match.similarity:.3f}`"
                )

                st.caption(movie.overview)
                st.divider()
        else:
            st.info("Please enter a search query.")
    else:
        st.info("Enter a title or a plot idea, then click Search.")


@st.fragment
def render_catalog_browser(movie_service: MovieService, embeddings, top_k: int, dataset_fingerprint: str):
    st.subheader("Browse the catalog")

    title_query = st.text_input("Find a movie by title", placeholder="Start typing a movie title...", key="search_movie_search")

    matching_movies = get_cached_title_search(movie_service, title_query.strip(), 20, dataset_fingerprint)
    selected = None

    if matching_movies:
        movies_by_id = {movie.id: movie for movie in matching_movies}

        selected_id = st.selectbox(
            "Matching movies",
            options=list(movies_by_id.keys()),
            format_func=lambda movie_id: (
                f"{movies_by_id[movie_id].title} "
                f"({movies_by_id[movie_id].year})"
            ),
            key="catalog_movie_selection"
        )

        selected = movies_by_id[selected_id]

    elif title_query.strip():
        st.info("No matching movie titles found.")
    else:
        st.info("Enter a movie title to browse the catalog.")

    if selected is not None:
        st.markdown(f"### {selected.title} ({selected.year})")

        st.write(selected.overview)

        st.write(
            f"**Genres:** "
            f"{', '.join(selected.genres) or 'Unknown'}"
        )

        st.write(
            f"**Director:** "
            f"{selected.director or 'Unknown'}"
        )

        st.write(
            f"**Cast:** "
            f"{', '.join(selected.actors) or 'Unknown'}"
        )

        results = get_cached_similar_movies(selected.id, top_k, dataset_fingerprint, movie_service, embeddings)

        st.markdown("**Movies with similar descriptions**")

        if not results:
            st.info("No similar movies found.")
        else:
            for match in results:
                st.write(
                    f"- **{match.movie.title}** "
                    f"— similarity `{match.similarity:.3f}`"
                )


def render_search_page(movie_service, embeddings, model, top_k: int, dataset_fingerprint: str):
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        render_semantic_search(movie_service, embeddings, model, top_k, dataset_fingerprint)

    with right:
        render_catalog_browser(movie_service, embeddings, top_k, dataset_fingerprint)
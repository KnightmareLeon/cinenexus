
import streamlit as st

from services.movie_service import MovieService
from services.query_cache import (get_cached_semantic_search, get_cached_title_search, get_cached_similar_movies, get_cached_tfidf_search)


def render_matches(results):
    if not results:
        st.info("No matching movies found. Try different words for TF-IDF.")
    for rank, match in enumerate(results, 1):
        movie = match.movie
        st.markdown(f"**{rank}. {movie.title} ({movie.year})** · similarity `{match.similarity:.3f}`")
        st.caption(f"Director: {getattr(movie, 'director', None) or 'Unknown'}")
        st.caption(movie.overview or "No description available.")
        st.divider()

@st.fragment
def render_semantic_search(movie_service: MovieService, embeddings, model, top_k: int, dataset_fingerprint: str, tfidf_index):
    st.subheader("Find a movie")

    with st.form("semantic_search_form"):
        query = st.text_input("Search by title or describe what you want to watch", placeholder="e.g., a space adventure about family")

        submitted = st.form_submit_button("Search")

    if submitted:
        if query.strip():
            results = {}
            try:
                with st.spinner("Searching movies…"):
                    results["SBERT"] = get_cached_semantic_search(query.strip(), top_k, dataset_fingerprint, movie_service, embeddings, model)
                    results["TF-IDF"] = get_cached_tfidf_search(query.strip(), top_k, dataset_fingerprint, movie_service, tfidf_index)
            except ValueError as exc:
                st.error(f"Could not run search: {exc}")
                return
            st.session_state.movie_search_result = (dataset_fingerprint, top_k, query.strip(), results)
        else:
            st.info("Please enter a search query.")
            return

    saved = st.session_state.get("movie_search_result")
    if (saved and saved[:2] == (dataset_fingerprint, top_k)
            and set(saved[3]) == {"SBERT", "TF-IDF"}):
        st.caption(f'Results for “{saved[2]}” · up to {top_k} per method. Similarity is not an accuracy score.')
        columns = st.columns(len(saved[3]))
        for column, (name, matches) in zip(columns, saved[3].items()):
            with column:
                st.markdown(f"**{name} matches**")
                render_matches(matches)
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


def render_search_page(movie_service, embeddings, model, top_k: int, dataset_fingerprint: str, tfidf_index):
    render_semantic_search(movie_service, embeddings, model, top_k, dataset_fingerprint, tfidf_index)
    with st.expander("Browse the catalog and find similar movies"):
        render_catalog_browser(movie_service, embeddings, top_k, dataset_fingerprint)

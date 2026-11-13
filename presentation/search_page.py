import streamlit as st

from services.movie_service import MovieService
from services.query_cache import (
    get_cached_semantic_search,
    get_cached_title_search,
    get_cached_similar_movies,
    get_cached_tfidf_search,
)


def _format_year(year_val) -> str:
    """Strips trailing .0 from float-formatted years and handles missing values."""
    if not year_val:
        return "N/A"
    s = str(year_val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s if s else "N/A"


def render_matches(results):
    if not results:
        st.caption("No matching movies found.")
        return

    for rank, match in enumerate(results, 1):
        movie = match.movie
        year_str = _format_year(movie.year)
        director_str = getattr(movie, "director", None) or "Unknown"
        genres_list = getattr(movie, "genres", None) or []
        genres_str = ", ".join(genres_list[:3])
        actors_list = getattr(movie, "actors", None) or []
        actors_str = ", ".join(actors_list[:4])

        meta_parts = [f"Director: {director_str}"]
        if genres_str:
            meta_parts.append(genres_str)

        with st.container(border=True):
            col_title, col_score = st.columns([4, 1])
            with col_title:
                st.markdown(f"**{rank}. {movie.title}** ({year_str})")
            with col_score:
                st.markdown(f"`{match.similarity:.3f}`")

            st.caption(" | ".join(meta_parts))
            if actors_str:
                st.caption(f"Cast: {actors_str}")
            st.write(movie.overview or "No description available.")


@st.fragment
def render_semantic_search(
    movie_service: MovieService,
    embeddings,
    model,
    top_k: int,
    dataset_fingerprint: str,
    tfidf_index,
):
    st.subheader("Find a movie")

    with st.form("semantic_search_form"):
        query = st.text_input(
            "Search by title, plot premise, actor, director, or genre",
            placeholder="e.g., Christopher Nolan sci-fi movie about entering dreams",
        )
        submitted = st.form_submit_button("Search")

    if submitted:
        if query.strip():
            results = {}
            try:
                with st.spinner("Searching catalog..."):
                    results["SBERT"] = get_cached_semantic_search(
                        query.strip(),
                        top_k,
                        dataset_fingerprint,
                        movie_service,
                        embeddings,
                        model,
                    )
                    results["TF-IDF"] = get_cached_tfidf_search(
                        query.strip(),
                        top_k,
                        dataset_fingerprint,
                        movie_service,
                        tfidf_index,
                    )
            except ValueError as exc:
                st.error(f"Could not run search: {exc}")
                return
            st.session_state.movie_search_result = (
                dataset_fingerprint,
                top_k,
                query.strip(),
                results,
            )
        else:
            st.info("Please enter a search query.")
            return

    saved = st.session_state.get("movie_search_result")
    if (
        saved
        and saved[:2] == (dataset_fingerprint, top_k)
        and set(saved[3]) == {"SBERT", "TF-IDF"}
    ):
        st.caption(
            f'Showing top {top_k} results per method for "{saved[2]}".'
        )
        col_sbert, col_tfidf = st.columns(2)
        with col_sbert:
            st.markdown("##### SBERT (Hybrid Semantic)")
            render_matches(saved[3]["SBERT"])
        with col_tfidf:
            st.markdown("##### TF-IDF (Lexical)")
            render_matches(saved[3]["TF-IDF"])
    else:
        st.caption("Enter a title, plot setup, or cast/director query above, then click Search.")


@st.fragment
def render_catalog_browser(
    movie_service: MovieService,
    embeddings,
    top_k: int,
    dataset_fingerprint: str,
):
    st.subheader("Browse the catalog")

    title_query = st.text_input(
        "Find a movie by title",
        placeholder="Start typing a movie title...",
        key="search_movie_search",
    )

    matching_movies = get_cached_title_search(
        movie_service, title_query.strip(), 20, dataset_fingerprint
    )
    selected = None

    if matching_movies:
        movies_by_id = {movie.id: movie for movie in matching_movies}

        selected_id = st.selectbox(
            "Matching movies",
            options=list(movies_by_id.keys()),
            format_func=lambda movie_id: (
                f"{movies_by_id[movie_id].title} "
                f"({_format_year(movies_by_id[movie_id].year)})"
            ),
            key="catalog_movie_selection",
        )

        selected = movies_by_id[selected_id]

    elif title_query.strip():
        st.info("No matching movie titles found.")
    else:
        st.caption("Enter a movie title above to inspect its metadata and similar films.")

    if selected is not None:
        with st.container(border=True):
            st.markdown(f"#### {selected.title} ({_format_year(selected.year)})")
            st.write(selected.overview or "No description available.")

            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.caption(f"**Director:** {selected.director or 'Unknown'}")
                st.caption(f"**Genres:** {', '.join(selected.genres) or 'Unknown'}")
            with meta_col2:
                cast_preview = ", ".join(selected.actors[:6]) if selected.actors else "Unknown"
                st.caption(f"**Cast:** {cast_preview}")

        results = get_cached_similar_movies(
            selected.id, top_k, dataset_fingerprint, movie_service, embeddings
        )

        st.markdown("##### Movies with similar descriptions & shared metadata")

        if not results:
            st.info("No similar movies found.")
        else:
            render_matches(results)


def render_search_page(
    movie_service,
    embeddings,
    model,
    top_k: int,
    dataset_fingerprint: str,
    tfidf_index,
):
    render_semantic_search(
        movie_service, embeddings, model, top_k, dataset_fingerprint, tfidf_index
    )
    st.write("")
    with st.expander("Browse the catalog and find similar movies"):
        render_catalog_browser(
            movie_service, embeddings, top_k, dataset_fingerprint
        )
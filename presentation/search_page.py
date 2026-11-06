import bisect

import streamlit as st

from services.recommendation_service import (search_movies, get_similar_movies)

@st.cache_resource
def build_movie_index(_movies):
    movie_by_id = {movie.id: movie for movie in _movies}

    title_index = sorted((movie.title.casefold(), movie.id, movie.title) for movie in _movies if movie.title)

    return movie_by_id, title_index


def find_movies_by_title(title_index, query, limit=20):
    query = query.strip().casefold()

    if not query:
        return []

    start = bisect.bisect_left(title_index, (query, "", ""))
    matches = []

    for index in range(start, len(title_index)):
        normalized_title, movie_id, _ = title_index[index]

        if not normalized_title.startswith(query):
            break

        matches.append(movie_id)

        if len(matches) >= limit:
            break

    return matches


@st.fragment
def render_semantic_search(movies, embeddings, model, top_k):
    st.subheader("Find a movie")

    with st.form("semantic_search_form"):
        query = st.text_input("Search by title or describe what you want to watch", placeholder="e.g., a space adventure about family")

        submitted = st.form_submit_button("Search")

    if submitted:
        if query.strip():
            results = search_movies(query, movies, embeddings, model, top_k)

            st.markdown("**Best semantic matches**")

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
def render_catalog_browser(movies, embeddings, top_k, movie_by_id, title_index):
    st.subheader("Browse the catalog")

    title_query = st.text_input("Find a movie by title", placeholder="Start typing a movie title...", key="search_movie_search")

    matching_ids = find_movies_by_title(title_index, title_query, limit=20)

    selected = None

    if matching_ids:
        selected_id = st.selectbox(
            "Matching movies",
            options=matching_ids,
            format_func=lambda movie_id: (
                f"{movie_by_id[movie_id].title} "
                f"({movie_by_id[movie_id].year})"
            ),
            key="catalog_movie_selection"
        )

        selected = movie_by_id[selected_id]

    elif title_query.strip():
        st.info("No matching movie titles found.")
    else:
        st.info("Enter a movie title to browse the catalog.")

    if selected is not None:
        st.markdown(f"### {selected.title} ({selected.year})")
        st.write(selected.overview)
        st.write(f"**Genres:** {selected.genres}")
        st.write(f"**Director:** {selected.director or 'Unknown'}")
        st.write(f"**Cast:** {selected.actors or 'Unknown'}")

        results = get_similar_movies(
            selected.id, movies, embeddings, top_k
        )

        st.markdown("**Movies with similar descriptions**")

        for match in results:
            st.write(
                f"- **{match.movie.title}** "
                f"— similarity `{match.similarity:.3f}`"
            )


def render_search_page(movies, embeddings, model, top_k):
    movie_by_id, title_index = build_movie_index(movies)

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        render_semantic_search(movies, embeddings, model, top_k)

    with right:
        render_catalog_browser(movies, embeddings, top_k,movie_by_id, title_index)

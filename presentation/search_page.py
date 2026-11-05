import streamlit as st
from services.recommendation_service import (search_movies, get_similar_movies)
from services.movie_service import get_movie_by_id

def render_search_page(movies, embeddings, model, top_k):
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.subheader("Find a movie")

        query = st.text_input(
            "Search by title or describe what you want to watch",
            placeholder="e.g., a space adventure about family"
        )

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
            st.info("Enter a title or a plot idea to find relevant movies.")

    with right:
        st.subheader("Browse the catalog")

        selected_id = st.selectbox(
            "Select a movie",
            options=[movie.id for movie in movies],
            format_func=lambda movie_id: next(
                movie.title
                for movie in movies
                if movie.id == movie_id
            )
        )

        selected = get_movie_by_id(movies, selected_id)

        if selected is not None:
            st.markdown(f"### {selected.title} ({selected.year})")
            st.write(selected.overview)
            st.write(f"**Genres:** {selected.genres}")
            st.write(f"**Director:** {selected.director or 'Unknown'}")
            st.write(f"**Cast:** {selected.actors or 'Unknown'}")

        results = get_similar_movies(selected.id, movies, embeddings, top_k)

        st.markdown("**Movies with similar descriptions**")

        for match in results:
            st.write(
                f"- **{match.movie.title}** "
                f"— similarity `{match.similarity:.3f}`"
            )
import streamlit as st


def render_dataset_page(movies):
    st.subheader("Loaded movie records")

    st.dataframe(
        movies[["title", "year", "genres", "director", "actors", "overview"]],
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "Download current dataset as CSV",
        data=movies.drop(
            columns=["search_text"],
            errors="ignore"
        ).to_csv(index=False),
        file_name="cinegraph_movies.csv",
        mime="text/csv"
    )
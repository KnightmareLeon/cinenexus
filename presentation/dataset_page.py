import math

import pandas as pd
import streamlit as st

from services.movie_service import MovieService
from services.dataset_export_service import DATASET_COLUMNS, movie_to_dataset_row, movies_to_csv
from services.query_cache import get_cached_catalog_csv, get_cached_movies_page

def previous_page():
    st.session_state.dataset_page_number -= 1
    st.session_state.dataset_page_input = st.session_state.dataset_page_number

def next_page():
    st.session_state.dataset_page_number += 1
    st.session_state.dataset_page_input = st.session_state.dataset_page_number

@st.fragment
def render_dataset_page(movie_service: MovieService, dataset_fingerprint: str):
    st.subheader("Movie Dataset")

    total_records = movie_service.get_movie_count()

    if total_records == 0:
        st.info("No movie records found in the database.")
        return

    page_size = 50
    total_pages = math.ceil(total_records / page_size)

    if "dataset_page_number" not in st.session_state:
        st.session_state.dataset_page_number = 1

    if "dataset_page_input" not in st.session_state:
        st.session_state.dataset_page_input = st.session_state.dataset_page_number

    current_page = min(max(1, st.session_state.dataset_page_number), total_pages)

    st.session_state.dataset_page_number = current_page

    movies = get_cached_movies_page(movie_service, current_page, page_size, dataset_fingerprint)

    start = (current_page - 1) * page_size
    end = start + len(movies)

    st.caption(
        f"Showing {start + 1:,}-{end:,} "
        f"of {total_records:,} movies"
    )

    dataframe = pd.DataFrame(
        [movie_to_dataset_row(movie) for movie in movies], columns=DATASET_COLUMNS
    )

    st.dataframe(dataframe, width="stretch", hide_index=True)

    st.download_button(
        "Download current page (CSV)",
        icon=":material/download:",
        data=movies_to_csv(movies),
        file_name=f"cinenexus-movies-page-{current_page}.csv",
        mime="text/csv",
        key=f"dataset_page_csv_{dataset_fingerprint}_{current_page}",
    )

    st.caption(
        f"The full catalog export includes all {total_records:,} movies. "
        "Preparing a large catalog may take some time."
    )
    catalog_csv = None
    if st.button("Prepare full catalog CSV", key="dataset_prepare_csv"):
        # Store only the dataset identity; the export bytes live in the shared cache.
        catalog_csv = get_cached_catalog_csv(movie_service, dataset_fingerprint)
        st.session_state.dataset_export_fingerprint = dataset_fingerprint

    if st.session_state.get("dataset_export_fingerprint") == dataset_fingerprint:
        if catalog_csv is None:
            catalog_csv = get_cached_catalog_csv(movie_service, dataset_fingerprint)
        st.download_button(
            "Download full catalog (CSV)",
            icon=":material/download:",
            data=catalog_csv,
            file_name="cinenexus-movies.csv",
            mime="text/csv",
            key=f"dataset_catalog_csv_{dataset_fingerprint}",
        )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.button("← Previous", disabled=current_page == 1, key="dataset_previous", on_click=previous_page)

    with col2:
        page_col, total_col = st.columns([2, 1])

        with page_col:
            selected_page = st.number_input(
                "Go to page",
                min_value=1,
                max_value=total_pages,
                step=1,
                key="dataset_page_input",
                label_visibility="collapsed"
            )

        with total_col:
            st.markdown(
                f"<div style='padding-top: 8px;'>/ {total_pages:,}</div>",
                unsafe_allow_html=True
            )

        if selected_page != current_page:
            st.session_state.dataset_page_number = selected_page
            st.rerun(scope="fragment")

    with col3:
        st.button("Next →", disabled=current_page == total_pages, key="dataset_next", on_click=next_page)

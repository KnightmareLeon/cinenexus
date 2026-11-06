import streamlit as st
import math

@st.fragment
def render_dataset_page(movies):
    st.subheader("Loaded movie records")

    page_size = 50
    total_records = len(movies)
    total_pages = max(1, math.ceil(total_records / page_size))

    if "dataset_page_number" not in st.session_state:
        st.session_state.dataset_page_number = 1

    current_page = st.session_state.dataset_page_number

    current_page = min(max(current_page, 1), total_pages)
    st.session_state.dataset_page_number = current_page

    start = (current_page - 1) * page_size
    end = min(start + page_size, total_records)

    st.caption(f"Showing {start + 1:,}-{end:,} of {total_records:,} movies")

    columns = ["title", "year", "genres", "director", "actors", "overview"]

    st.dataframe(movies.iloc[start:end][columns], width="stretch", hide_index=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("← Previous", disabled=current_page == 1, width="stretch"):
            st.session_state.dataset_page_number -= 1

    with col2:
        st.write(f"Page {current_page:,} of {total_pages:,}")

    with col3:
        if st.button("Next →", disabled=current_page == total_pages, width="stretch"):
            st.session_state.dataset_page_number += 1
import streamlit as st

from ml.tfidf import build_tfidf_matrix, compute_tfidf_scores


@st.cache_data(show_spinner=False)
def get_tfidf_embeddings(texts, description="TF-IDF"):
    """Fit TF-IDF on texts and cache the vectorizer + matrix.

    Args:
        texts: Iterable of strings (one per movie).
        description: Human-readable label for logging/debugging.

    Returns:
        dict with keys:
            - vectorizer: fitted TfidfVectorizer
            - matrix: sparse TF-IDF matrix (n_documents, n_features)
            - description: label passed in
    """
    vectorizer, matrix = build_tfidf_matrix(
        texts, lowercase=True, strip_accents="unicode", stop_words="english"
    )
    return {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "description": description,
    }


def compute_tfidf_scores_cached(cached, query_texts):
    """Compute TF-IDF scores using the cached vectorizer/matrix.

    Args:
        cached: dict returned by get_tfidf_embeddings.
        query_texts: Iterable of query strings.

    Returns:
        1D numpy.ndarray of shape (n_query_texts,) of cosine similarity scores.
    """
    return compute_tfidf_scores(
        cached["vectorizer"], cached["matrix"], query_texts
    )
import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from services.evaluation_service import TEST_QUERIES, run_benchmark
from services.query_cache import get_tfidf_index


SCORE_COLUMNS = ["Accuracy@1", "Precision@k", "Recall@k", "F1@k", "Time (ms)"]


def _show_scores(frame):
    st.dataframe(frame.style.format({
        **{column: "{:.1%}" for column in SCORE_COLUMNS[:-1]},
        "Time (ms)": "{:.1f}",
    }), width="stretch")


def _comparison_frame(scores):
    """Show the expected film's status beside the method's first prediction."""
    frame = scores.copy()
    frame["Expected movie"] = frame["Expected movie"] + " · " + frame["Expected director"]
    frame["Expected film status"] = [
        f"{movie}: found at rank {rank}" if str(rank).isdigit() else f"{movie}: {rank.lower()}"
        for movie, rank in zip(frame["Expected movie"], frame["Target rank"])
    ]
    frame["Predicted film (rank 1)"] = [
        "No results" if movie == "No results" else f"{movie} · {director}"
        for movie, director in zip(frame["First result"], frame["First result director"])
    ]
    return frame.set_index(["Query", "Method"])[
        ["Expected film status", "Predicted film (rank 1)", *SCORE_COLUMNS]
    ]


@st.fragment
def render_metrics_page(movie_service, embeddings, model, top_k, dataset_fingerprint):
    test_cases = [asdict(case) for case in TEST_QUERIES]
    st.subheader("Search metrics: SBERT vs TF-IDF")
    st.write("Compare both methods on preset test queries with saved target movies. No manual labeling needed.")
    st.caption("These ten exact movie titles were each verified to occur only once in the current catalog, across all years. Each has one saved expected movie. Similar titles and sequels can still appear in search results.")
    st.caption(f"Each method searches the full dataset and returns up to {top_k} movies (k = {top_k}).")
    with st.expander("Test queries and expected movies"):
        st.dataframe(pd.DataFrame([
            {"Query": case.query, "Expected movie": f"{case.title} ({case.year})"}
            for case in TEST_QUERIES
        ]), hide_index=True, width="stretch")

    if st.button("Run test queries", type="primary"):
        # Clear an earlier run so a failed rerun cannot leave old scores visible.
        st.session_state.pop("metrics_benchmark", None)
        try:
            index = get_tfidf_index(dataset_fingerprint, movie_service)
            with st.spinner("Testing both search methods…"):
                report = run_benchmark(top_k, movie_service, embeddings, model, index)
        except ValueError as exc:
            st.error(f"Could not run tests: {exc}")
            return
        st.session_state.metrics_benchmark = {
            **report, "top_k": top_k, "dataset_fingerprint": dataset_fingerprint,
            "test_cases": test_cases,
            "scope": "Fixed known-movie lookup tests; each query has one expected movie. Not general relevance evaluation.",
            "accuracy_definition": "Fraction of test queries whose first result is the expected movie.",
            "f1_definition": "Per-query F1 of returned movies against the saved target; summary is the unweighted query mean.",
            "timing_definition": "One uncached search per query and method; excludes model loading, index construction and scoring.",
        }

    run = st.session_state.get("metrics_benchmark")
    if not run:
        st.info("Run the test queries to see accuracy, F1, and search time for each method.")
        return
    if (run["dataset_fingerprint"] != dataset_fingerprint or run["top_k"] != top_k
            or "accuracy_definition" not in run or run.get("test_cases") != test_cases):
        st.info("The test queries, dataset, or result count changed. Run the test queries again to refresh the scores.")
        return
    if run["skipped"]:
        st.warning(f'{len(run["skipped"])} test queries skipped because their expected movies are missing or changed.')
        with st.expander("Skipped tests"):
            st.dataframe(pd.DataFrame(run["skipped"]), hide_index=True, width="stretch")
    if not run["metrics"]:
        st.info("No test targets are available in this dataset.")
        return

    scores = pd.DataFrame(run["metrics"])
    summary = scores.groupby("Method", sort=False)[SCORE_COLUMNS].mean()
    st.markdown("**Average scores by method**")
    _show_scores(summary)
    st.caption(f'{len(scores) // 2} completed queries · unweighted mean across queries · k = {top_k}.')
    if (scores["Accuracy@1"] == 0).all():
        st.info("None of the expected movies ranked first. Check Expected film status below: an expected movie at rank 2 or later still earns recall and F1 credit. A different movie with the same title does not count as the saved target.")
    st.markdown("**Scores per test query**")
    _show_scores(_comparison_frame(scores))
    st.caption('“Expected film status” describes the saved correct answer. “Predicted film (rank 1)” is the method’s actual first result. Open “Inspect returned movies” to see all returned predictions.')
    st.caption("Accuracy@1: expected movie ranked first (100% or 0% per query). F1@k: precision/recall balance against the saved target. These tests measure finding a specific movie, not general recommendation relevance.")
    with st.expander("Inspect returned movies"):
        selected_query = st.selectbox("Test query", scores["Query"].drop_duplicates().tolist())
        for result in run["results"]:
            if result["query"] != selected_query:
                continue
            row = scores[(scores["Query"] == selected_query) & (scores["Method"] == result["method"])].iloc[0]
            expected = f'{row["Expected movie"]} · {row["Expected director"]}'
            st.markdown(f'**{result["method"]}**')
            st.write(f'Expected: {expected}. Expected movie rank: {row["Target rank"]}.')
            st.caption(f'Expected movie ID: {result["expected_movie_id"]}')
            if not result["matches"]:
                st.write("No movies returned.")
                continue
            st.dataframe(pd.DataFrame([
                {"Rank": rank, "Title": match["title"], "Year": match.get("year", ""),
                 "Director": match.get("director", "Unknown"),
                 "Movie ID": match["movie_id"],
                 "Expected target": match["movie_id"] == result["expected_movie_id"]}
                for rank, match in enumerate(result["matches"], start=1)
            ]), hide_index=True, width="stretch")
    with st.expander("How to read the metrics"):
        st.write("Precision is the fraction of returned movies matching the target; recall is 100% if the target is returned, otherwise 0%. Empty results score zero. F1 = 2 × precision × recall / (precision + recall).")
        st.write(f"Each query has one target. When exactly {top_k} movies are returned, even a successful lookup has F1 = {2 / (top_k + 1):.1%}, because the other results are not that target.")
        st.write("Times measure one uncached search per query, including query encoding, ranking, and movie retrieval. Model loading and TF-IDF index building are excluded. They are not repeated timing trials.")
    report = {**run, "summary": summary.reset_index().to_dict(orient="records")}
    st.download_button("Download test metrics (JSON)", json.dumps(report, indent=2),
                       file_name="search-test-metrics.json", mime="application/json")

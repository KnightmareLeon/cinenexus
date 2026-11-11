import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from services.evaluation_service import TEST_QUERIES, run_benchmark

SCORE_COLUMNS = ["Accuracy@1", "Hit@5", "Time (ms)"]

def _show_scores(frame, hide_index=False):
    st.dataframe(frame.style.format({
        **{column: "{:.1%}" for column in SCORE_COLUMNS[:-1]},
        "Time (ms)": "{:.1f}",
    }), width="stretch", hide_index=hide_index)


def _comparison_frame(scores):
    """Show the expected film's status beside the method's first prediction."""
    frame = scores.copy()
    frame["Expected movie"] = frame["Expected movie"] + " · " + frame["Expected director"]
    frame["Predicted movie (rank 1)"] = [
        "No results" if movie == "No results" else f"{movie} · {director}"
        for movie, director in zip(frame["First result"], frame["First result director"])
    ]
    return frame[["Query", "Expected movie", "Predicted movie (rank 1)", "Method", *SCORE_COLUMNS]]


@st.fragment
def render_metrics_page(movie_service, embeddings, model, dataset_fingerprint, tfidf_index):
    top_k = 5
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
            with st.spinner("Testing both search methods…"):
                report = run_benchmark(top_k, movie_service, embeddings, model, tfidf_index)
        except ValueError as exc:
            st.error(f"Could not run tests: {exc}")
            return
        st.session_state.metrics_benchmark = {
            **report, "top_k": top_k, "dataset_fingerprint": dataset_fingerprint,
            "test_cases": test_cases,
            "scope": "Fixed known-movie lookup tests; each query has one expected movie. Not general relevance evaluation.",
            "accuracy_definition": "Fraction of test queries whose first result is the expected movie.",
            "hit_definition": "Fraction of test queries whose expected movie appears in the first five results.",
            "timing_definition": "One uncached search per query and method; excludes model loading, index construction and scoring.",
        }

    run = st.session_state.get("metrics_benchmark")
    if not run:
        st.info("Run the test queries to see Accuracy@1, Hit@5, and search time for each method.")
        return
    if (run["dataset_fingerprint"] != dataset_fingerprint or run["top_k"] != top_k
            or "hit_definition" not in run or run.get("test_cases") != test_cases):
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
        st.info("None of the expected movies ranked first. An expected movie at ranks 2–5 still earns Hit@5 credit. A different movie with the same title does not count as the saved target.")
    st.markdown("**Scores per test query**")
    _show_scores(_comparison_frame(scores), hide_index=True)
    st.caption('“Expected movie” is the saved correct answer. “Predicted movie (rank 1)” is the method’s actual first result. Open “Inspect returned movies” to see all returned predictions.')
    st.caption("Accuracy@1: expected movie ranked first (100% or 0% per query). Hit@5: expected movie appears in the first five results (100% or 0% per query). These tests measure finding a specific movie, not general recommendation relevance.")
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
    report = {**run, "summary": summary.reset_index().to_dict(orient="records")}
    st.download_button(
        "Download per-query metrics (CSV)",
        _comparison_frame(scores).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"search-test-metrics-k{top_k}.csv",
        mime="text/csv",
    )
    st.download_button("Download test metrics (JSON)", json.dumps(report, indent=2),
                       file_name="search-test-metrics.json", mime="application/json")

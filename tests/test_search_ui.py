import unittest

from streamlit.testing.v1 import AppTest


PRELUDE = '''
import numpy as np
import streamlit as st
from tests.test_search_evaluation import SearchFixture, FakeModel
service = SearchFixture()
embeddings = np.array([[1., 0.], [0., 1.], [.8, .6], [0., 1.]])
'''


class SearchUITests(unittest.TestCase):
    def test_comparison_and_empty_tfidf(self):
        at = AppTest.from_string(PRELUDE + '''
from presentation.search_page import render_search_page
from ml.tfidf import TfidfIndex
render_search_page(service, embeddings, FakeModel(), 3, "ui-search-fixture", TfidfIndex(service.texts))
''', default_timeout=10).run()
        at.text_input[0].set_value("spaceship")
        self.assertEqual(len(at.radio), 0)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(set(at.session_state.movie_search_result[3]), {"SBERT", "TF-IDF"})
        at.run()
        self.assertTrue(any("SBERT matches" in element.value for element in at.markdown))
        at.text_input[0].set_value("zzzzunknown")
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.movie_search_result[3]["TF-IDF"], [])

    def test_same_title_results_show_director_and_missing_description(self):
        at = AppTest.from_string('''
from types import SimpleNamespace
from presentation.search_page import render_matches
matches = [SimpleNamespace(movie=SimpleNamespace(
    title="Barbie", year="2023.0", overview=overview, director=director,
), similarity=score) for director, overview, score in (
    ("Mohammed Al-Hamali", "", 1.0),
    ("Greta Gerwig", "Barbie and Ken visit the real world.", 0.7),
)]
render_matches(matches)
''').run()
        self.assertFalse(at.exception)
        captions = [element.value for element in at.caption]
        self.assertIn("Director: Mohammed Al-Hamali", captions)
        self.assertIn("Director: Greta Gerwig", captions)
        self.assertIn("No description available.", captions)

    def metrics_app(self):
        return AppTest.from_string(PRELUDE + '''
from unittest.mock import patch
from services.evaluation_service import SearchTestCase, run_benchmark
from presentation.metrics_page import render_metrics_page
from ml.tfidf import TfidfIndex
cases = (SearchTestCase("spaceship", "0", "Movie 0", "2024"),
         SearchTestCase("chef", "1", "Movie 1", "2024"))
def fixture_benchmark(*args):
    return run_benchmark(*args, cases=cases)
with patch("presentation.metrics_page.run_benchmark", side_effect=fixture_benchmark):
    render_metrics_page(service, embeddings, FakeModel(),
                        st.session_state.get("test_fingerprint", "ui-metrics-fixture"), TfidfIndex(service.texts))
''').run()

    def test_metrics_run_without_labels_and_show_averages(self):
        at = self.metrics_app()
        self.assertEqual(len(at.text_input), 0)
        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.dataframe), 5)
        summary = at.dataframe[1].value
        self.assertEqual(summary.loc["TF-IDF", "Accuracy@1"], 1)
        self.assertEqual(summary.loc["TF-IDF", "Hit@5"], 1)
        self.assertEqual(summary.loc["SBERT", "Accuracy@1"], 0.5)
        self.assertEqual(summary.loc["SBERT", "Hit@5"], 1)
        self.assertEqual(len(at.session_state.metrics_benchmark["metrics"]), 4)
        self.assertEqual(at.dataframe[2].value.iloc[0]["Expected movie"], "Movie 0 (2024) · Unknown")
        self.assertEqual(at.dataframe[3].value.iloc[0]["Expected target"], True)
        at.selectbox[0].set_value("chef").run()
        self.assertFalse(at.exception)
        self.assertTrue(at.dataframe[3].value["Expected target"].any())
        self.assertTrue(at.dataframe[4].value.iloc[0]["Expected target"])
        at.run()
        self.assertEqual(len(at.dataframe), 5)
        self.assertFalse(at.exception)

    def test_metrics_invalidated_by_settings_and_dataset(self):
        at = self.metrics_app()
        at.button[0].click().run()
        at.session_state.metrics_benchmark.pop("hit_definition")
        at.run()
        self.assertEqual(len(at.dataframe), 1)
        self.assertTrue(any("changed" in item.value for item in at.info))

        at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.metrics_benchmark["top_k"], 5)
        at.session_state.metrics_benchmark["test_cases"] = []
        at.run()
        self.assertEqual(len(at.dataframe), 1)
        self.assertTrue(any("test queries" in item.value for item in at.info))
        at.button[0].click().run()
        self.assertFalse(at.exception)
        at.session_state["test_fingerprint"] = "updated-dataset"
        at.run()
        self.assertEqual(len(at.dataframe), 1)
        self.assertTrue(any("changed" in item.value for item in at.info))

    def test_metrics_show_expected_and_predicted_movies(self):
        at = AppTest.from_string('''
import pandas as pd
from presentation.metrics_page import _comparison_frame, _show_scores
scores = pd.DataFrame([{
    "Query": "Barbie", "Method": "TF-IDF", "Expected movie": "Barbie (2023)",
    "Expected director": "Greta Gerwig", "First result": "Barbie (2023)",
    "First result director": "Mohammed Al-Hamali", "Target rank": "Not in top 10",
    "Accuracy@1": 0., "Hit@5": 0., "Time (ms)": 1.,
}])
_show_scores(_comparison_frame(scores))
''').run()
        self.assertFalse(at.exception)
        row = at.dataframe[0].value.iloc[0]
        self.assertEqual(row["Expected movie"], "Barbie (2023) · Greta Gerwig")
        self.assertEqual(row["Predicted movie (rank 1)"], "Barbie (2023) · Mohammed Al-Hamali")


if __name__ == "__main__":
    unittest.main()

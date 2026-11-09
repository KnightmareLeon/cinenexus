import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from ml.tfidf import TfidfIndex
from services.evaluation_service import SearchTestCase, compare_searches, evaluate_results, run_benchmark
from services.recommendation_service import search_movies_tfidf


class SearchFixture:
    texts = ["astronaut spaceship moon", "chef kitchen dinner", "astronaut moon journey", "romantic wedding"]

    def __init__(self):
        self.movies = [SimpleNamespace(
            id=str(i), title=f"Movie {i}", year="2024", overview=text,
        ) for i, text in enumerate(self.texts)]

    def iter_search_texts(self):
        return iter(self.texts)

    def get_movie_ids_by_embedding_indices(self, indices):
        return [str(i) for i in indices]

    def get_movies_by_ids(self, ids):
        return [self.movies[int(i)] for i in ids]

    def search_movies_by_title(self, query, limit=20):
        return []


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] for _ in texts], dtype=np.float32)


class SearchEvaluationTests(unittest.TestCase):
    def test_tfidf_ranking_and_mapping(self):
        service = SearchFixture()
        index = TfidfIndex(service.iter_search_texts())
        matches = search_movies_tfidf("spaceship", service, index, 5)
        self.assertEqual([match.movie.id for match in matches], ["0"])
        self.assertGreater(matches[0].similarity, 0)
        self.assertLessEqual(matches[0].similarity, 1)
        np.testing.assert_allclose(np.asarray(index.matrix.multiply(index.matrix).sum(axis=1)).ravel(), 1, rtol=1e-6)

    def test_tfidf_empty_unknown_and_ties(self):
        index = TfidfIndex(["space moon", "space moon", "kitchen chef"])
        for query in ("", "the and", "zzzzunknown"):
            self.assertEqual(index.search(query), ([], []))
        self.assertEqual(index.search("space", 10)[0], [0, 1])
        self.assertEqual(index.search("space", 1)[0], [0])
        self.assertEqual(index.search("space", 0), ([], []))

    def test_known_target_scores_and_ranking(self):
        scores = evaluate_results(["a", "b", "c"], ["a"])
        self.assertEqual(scores["Accuracy@1"], 1)
        self.assertAlmostEqual(scores["Precision@k"], 1 / 3)
        self.assertEqual(scores["Recall@k"], 1)
        self.assertEqual(scores["F1@k"], 0.5)
        second = evaluate_results(["b", "a", "c"], ["a"])
        self.assertEqual(second["Accuracy@1"], 0)
        self.assertEqual(second["F1@k"], 0.5)
        self.assertTrue(all(value == 1 for value in evaluate_results(["a", "a"], ["a"]).values()))

    def test_missing_target_and_empty_predictions_score_zero(self):
        for predictions in ([], ["b", "c"]):
            self.assertTrue(all(value == 0 for value in evaluate_results(predictions, ["a"]).values()))
        with self.assertRaises(ValueError):
            evaluate_results(["a"], [])

    def test_benchmark_scores_fixed_targets_and_skips_changed_movies(self):
        service = SearchFixture()
        cases = (
            SearchTestCase("spaceship", "0", "Movie 0", "2024"),
            SearchTestCase("chef", "1", "Movie 1", "2024"),
            SearchTestCase("wrong title", "2", "Different title", "2024"),
            SearchTestCase("wrong year", "3", "Movie 3", "1999"),
        )
        service.movies[0].year = "2024.0"
        report = run_benchmark(2, service, np.array([[1., 0.], [0., 1.], [.8, .6], [0., 1.]]),
                               FakeModel(), TfidfIndex(service.texts), cases)
        self.assertEqual(len(report["metrics"]), 4)
        self.assertEqual(len(report["skipped"]), 2)
        self.assertEqual(report["metrics"][0]["F1@k"], 2 / 3)
        self.assertEqual(report["metrics"][1]["F1@k"], 1)
        self.assertEqual(report["metrics"][2]["Accuracy@1"], 0)
        self.assertEqual(report["metrics"][3]["Accuracy@1"], 1)
        self.assertEqual(report["metrics"][0]["Target rank"], "1")
        self.assertEqual(report["metrics"][2]["Target rank"], "Not in top 2")
        self.assertEqual(report["metrics"][2]["First result"], "Movie 0 (2024)")
        with patch.object(service, "get_movies_by_ids", return_value=[]), patch("services.evaluation_service.compare_searches") as search:
            missing = run_benchmark(2, service, None, None, None, cases)
        search.assert_not_called()
        self.assertEqual(missing["metrics"], [])
        self.assertEqual(len(missing["skipped"]), 4)

    def test_comparison_runs_both_methods(self):
        service = SearchFixture()
        results = compare_searches("spaceship", 2, service, np.array([[1., 0.], [0., 1.], [.8, .6], [0., 1.]]), FakeModel(), TfidfIndex(service.texts))
        self.assertEqual([m.movie.id for m in results["SBERT"]["matches"]], ["0", "2"])
        self.assertEqual([m.movie.id for m in results["TF-IDF"]["matches"]], ["0"])
        self.assertTrue(all(result["elapsed_ms"] >= 0 for result in results.values()))

    def test_same_title_search_result_and_expected_movie_have_separate_ranks(self):
        service = SearchFixture()
        service.texts = ["Barbie and Ken travel to the real world", "Barbie"]
        service.movies = [SimpleNamespace(id=str(i), title="Barbie", year="2023.0",
                          director=director, overview=service.texts[i])
                          for i, director in enumerate(("Greta Gerwig", "Mohammed Al-Hamali"))]
        index = TfidfIndex(service.texts)
        embeddings = np.array([[0., 1.], [1., 0.]])
        cases = (SearchTestCase("Barbie", "0", "Barbie", "2023"),)
        direct = search_movies_tfidf("Barbie", service, index, 1)
        report = run_benchmark(1, service, embeddings, FakeModel(), index, cases)
        self.assertEqual([match.movie.id for match in direct], ["1"])
        self.assertEqual([match["movie_id"] for match in report["results"][1]["matches"]], ["1"])
        for row in report["metrics"]:
            self.assertEqual(row["Expected director"], "Greta Gerwig")
            self.assertEqual(row["First result director"], "Mohammed Al-Hamali")
            self.assertEqual(row["Target rank"], "Not in top 1")
            self.assertEqual(row["Accuracy@1"], 0)
        expanded = run_benchmark(2, service, embeddings, FakeModel(), index, cases)
        for row in expanded["metrics"]:
            self.assertEqual(row["Target rank"], "2")
            self.assertEqual(row["Accuracy@1"], 0)
            self.assertEqual(row["Recall@k"], 1)


if __name__ == "__main__":
    unittest.main()

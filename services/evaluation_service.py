from time import perf_counter
from dataclasses import dataclass

from services.recommendation_service import search_movies, search_movies_tfidf


def compare_searches(query, top_k, movie_service, embeddings, model, tfidf_index):
    """Measure uncached query execution, excluding model/index preparation."""
    results = {}
    for name, search in (
        ("SBERT", lambda: search_movies(query, movie_service, embeddings, model, top_k)),
        ("TF-IDF", lambda: search_movies_tfidf(query, movie_service, tfidf_index, top_k)),
    ):
        start = perf_counter()
        matches = search()
        results[name] = {"matches": matches, "elapsed_ms": (perf_counter() - start) * 1000}
    return results


@dataclass(frozen=True)
class SearchTestCase:
    query: str
    movie_id: str
    title: str
    year: str


# Exact title queries, each verified to occur once in the bundled catalog
# (case-insensitive, across all years). References are independent of search ranks.
TEST_QUERIES = (
    SearchTestCase("Inception", "1000018", "Inception", "2010"),
    SearchTestCase("Interstellar", "1000007", "Interstellar", "2014"),
    SearchTestCase("The Shawshank Redemption", "1000068", "The Shawshank Redemption", "1994"),
    SearchTestCase("Eternal Sunshine of the Spotless Mind", "1000029", "Eternal Sunshine of the Spotless Mind", "2004"),
    SearchTestCase("Everything Everywhere All at Once", "1000003", "Everything Everywhere All at Once", "2022"),
    SearchTestCase("The Grand Budapest Hotel", "1000024", "The Grand Budapest Hotel", "2014"),
    SearchTestCase("The Truman Show", "1000022", "The Truman Show", "1998"),
    SearchTestCase("Spirited Away", "1000030", "Spirited Away", "2001"),
    SearchTestCase("The Silence of the Lambs", "1000042", "The Silence of the Lambs", "1991"),
    SearchTestCase("The Curious Case of Benjamin Button", "1000424", "The Curious Case of Benjamin Button", "2008"),
)


def evaluate_results(retrieved_ids, expected_ids):
    """Score known-target retrieval; accuracy is top-1 correctness, not catalog TNs."""
    expected = set(expected_ids)
    if not expected:
        raise ValueError("At least one expected movie is required.")
    ranked = list(dict.fromkeys(retrieved_ids))
    return {
        "Accuracy@1": float(bool(ranked) and ranked[0] in expected),
        "Hit@5": float(bool(set(ranked[:5]) & expected)),
    }


def run_benchmark(top_k, movie_service, embeddings, model, tfidf_index, cases=TEST_QUERIES):
    """Run the same fixed queries against both methods on the full catalog."""
    if top_k < 1:
        raise ValueError("The result count must be positive.")
    movies = {movie.id: movie for movie in movie_service.get_movies_by_ids([case.movie_id for case in cases])}
    rows, results, skipped = [], [], []
    for case in cases:
        movie = movies.get(case.movie_id)
        if (movie is None or movie.title.casefold() != case.title.casefold()
                or str(movie.year).removesuffix(".0") != case.year):
            skipped.append({"query": case.query, "expected_movie_id": case.movie_id,
                            "reason": f"Expected {case.title} ({case.year}) is missing or changed."})
            continue
        comparison = compare_searches(case.query, top_k, movie_service, embeddings, model, tfidf_index)
        for name, result in comparison.items():
            matches = result["matches"]
            target_rank = next((rank for rank, match in enumerate(matches, start=1)
                                if match.movie.id == case.movie_id), None)
            first_movie = matches[0].movie if matches else None
            rows.append({
                "Query": case.query, "Expected movie": f"{case.title} ({case.year})", "Method": name,
                "Expected director": getattr(movie, "director", None) or "Unknown",
                "First result": (f"{first_movie.title} ({str(first_movie.year).removesuffix('.0')})"
                                 if first_movie else "No results"),
                "First result director": getattr(first_movie, "director", None) or "Unknown",
                "Target rank": str(target_rank) if target_rank else f"Not in top {top_k}",
                **evaluate_results([match.movie.id for match in matches], [case.movie_id]),
                "Time (ms)": result["elapsed_ms"],
            })
            results.append({"query": case.query, "method": name, "expected_movie_id": case.movie_id,
                            "matches": [{"movie_id": match.movie.id, "title": match.movie.title,
                                         "year": str(match.movie.year).removesuffix(".0"),
                                         "director": getattr(match.movie, "director", None) or "Unknown",
                                         "similarity": match.similarity} for match in matches]})
    return {"metrics": rows, "results": results, "skipped": skipped}

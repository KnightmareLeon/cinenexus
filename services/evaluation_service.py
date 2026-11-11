import csv
import random
import re
import unicodedata

from time import perf_counter
from dataclasses import dataclass

from services.recommendation_service import search_movies, search_movies_tfidf

from datetime import datetime
from pathlib import Path

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

## plot-test queries 
PLOT_TEST_QUERIES = (
    SearchTestCase(
        "A thief enters people's dreams to plant an idea.",
        "1000018", "Inception", "2010",
    ),
    SearchTestCase(
        "A dream infiltration mission becomes a dangerous heist.",
        "1000018", "Inception", "2010",
    ),
    SearchTestCase(
        "Explorers travel through a wormhole to find humanity a new home.",
        "1000007", "Interstellar", "2014",
    ),
    SearchTestCase(
        "Space travelers search for a habitable planet as Earth becomes unlivable.",
        "1000007", "Interstellar", "2014",
    ),
    SearchTestCase(
        "A man discovers that his entire life is a television show.",
        "1000022", "The Truman Show", "1998",
    ),
    SearchTestCase(
        "Someone gradually realizes his ordinary world has been staged.",
        "1000022", "The Truman Show", "1998",
    ),
    SearchTestCase(
        "A couple undergoes a procedure to erase memories of their relationship.",
        "1000029", "Eternal Sunshine of the Spotless Mind", "2004",
    ),
    SearchTestCase(
        "Former lovers try to forget each other through memory removal.",
        "1000029", "Eternal Sunshine of the Spotless Mind", "2004",
    ),
    SearchTestCase(
        "A man is born elderly and grows younger over time.",
        "1000424", "The Curious Case of Benjamin Button", "2008",
    ),
    SearchTestCase(
        "A person's life unfolds with aging running in reverse.",
        "1000424", "The Curious Case of Benjamin Button", "2008",
    ),
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

def normalize_movie_title(title):
    """Normalize titles for case- and punctuation-insensitive matching."""
    title = unicodedata.normalize("NFKC", str(title)).casefold()
    title = re.sub(r"[^\w\s]", " ", title)
    return " ".join(title.split())


def normalize_movie_year(year):
    """Normalize database years such as '2010.0' to '2010'."""
    if year is None:
        return ""

    value = str(year).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value

def normalize_director_name(name):
    """Normalize director names for case- and punctuation-insensitive matching."""
    name = unicodedata.normalize("NFKC", str(name)).casefold()
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split())

def parse_release_year(date_value):
    """Extract a year from the IMDb dataset's release date."""
    if not date_value:
        return None

    date_value = str(date_value).strip()

    formats = (
        "%m/%d/%Y",
        "%Y-%m-%d",
        "%Y",
    )

    for date_format in formats:
        try:
            return str(datetime.strptime(date_value, date_format).year)
        except ValueError:
            continue

    return None

def generate_imdb_test_cases(imdb_csv_path, movie_service, output_path="imdb_test_cases.csv", skipped_path="imdb_skipped_cases.csv", sample_size=500, random_state=42,):
    """
    Generate external evaluation cases from the IMDb CSV.

    Movies are matched using normalized title and release year.
    Ambiguous and unmatched records are excluded.
    """

    if sample_size < 1:
        raise ValueError("Sample size must be positive.")

    imdb_csv_path = Path(imdb_csv_path)
    output_path = Path(output_path)
    skipped_path = Path(skipped_path)

    matched_cases = []
    skipped = []

    seen_source_keys = set()
    title_search_cache = {}

    total_records = 0
    duplicate_source_records = 0

    with imdb_csv_path.open("r", encoding="utf-8-sig", newline="",) as file:

        reader = csv.DictReader(file)

        required_columns = {"names", "date_x", "overview"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"IMDb CSV must contain these columns: {required_columns}")

        for row in reader:
            total_records += 1

            title = (row.get("names") or "").strip()
            overview = (row.get("overview") or "").strip()
            year = parse_release_year(row.get("date_x"))

            if not title:
                skipped.append({"title": "", "year": year or "", "reason": "Missing title",})
                continue

            if not overview:
                skipped.append({"title": title, "year": year or "", "reason": "Missing overview",})
                continue

            if not year:
                skipped.append({"title": title, "year": "", "reason": "Invalid or missing release date",})
                continue

            normalized_title = normalize_movie_title(title)
            source_key = (normalized_title, year)

            if source_key in seen_source_keys:
                duplicate_source_records += 1
                continue

            seen_source_keys.add(source_key)

            if normalized_title not in title_search_cache:
                title_search_cache[normalized_title] = (movie_service.search_movies_by_title(query=title, limit=100,))

            candidates = title_search_cache[normalized_title]

            matches = [
                movie
                for movie in candidates
                if (
                    normalize_movie_title(movie.title) == normalized_title
                    and normalize_movie_year(movie.year) == year
                )
            ]

            if not matches:
                skipped.append({"title": title, "year": year, "reason": "No matching CineNexus movie",})
                continue

            if len(matches) > 1:
                skipped.append({"title": title, "year": year, "reason": "Multiple matching CineNexus movies",})
                continue

            movie = matches[0]

            matched_cases.append(SearchTestCase(query=overview, movie_id=str(movie.id), title=str(movie.title), year=year,))

    rng = random.Random(random_state)

    selected_cases = rng.sample(matched_cases, k=min(sample_size, len(matched_cases)))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="",) as file:

        writer = csv.DictWriter(file, fieldnames=["query", "movie_id", "title", "year"],)

        writer.writeheader()

        for case in selected_cases:
            writer.writerow({"query": case.query, "movie_id": case.movie_id, "title": case.title, "year": case.year,})

    skipped_path.parent.mkdir(parents=True, exist_ok=True)

    with skipped_path.open("w", encoding="utf-8", newline="",) as file:

        writer = csv.DictWriter(file, fieldnames=["title", "year", "reason"],)

        writer.writeheader()
        writer.writerows(skipped)

    return {
        "total_records": total_records,
        "matched_records": len(matched_cases),
        "selected_records": len(selected_cases),
        "skipped_records": len(skipped),
        "duplicate_source_records": duplicate_source_records,
        "test_cases": selected_cases,
        "output_path": str(output_path),
        "skipped_path": str(skipped_path),
    }


def load_imdb_test_cases(csv_path):
    """Load generated IMDb test cases into SearchTestCase objects."""

    cases = []

    with Path(csv_path).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            cases.append(
                SearchTestCase(
                    query=row["query"],
                    movie_id=row["movie_id"],
                    title=row["title"],
                    year=row["year"],
                )
            )

    return tuple(cases)
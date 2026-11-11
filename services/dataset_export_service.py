"""Convert catalog records to the columns shown on the dataset page."""

import csv
from collections.abc import Iterable
from io import BytesIO, TextIOWrapper

from domain.movie import Movie


DATASET_COLUMNS = ("Title", "Year", "Genres", "Director", "Actors", "Overview")


def movie_to_dataset_row(movie: Movie) -> dict[str, str]:
    return {
        "Title": movie.title,
        "Year": movie.year,
        "Genres": ", ".join(movie.genres),
        "Director": movie.director or "Unknown",
        "Actors": ", ".join(movie.actors),
        "Overview": movie.overview,
    }


def movies_to_csv(movies: Iterable[Movie]) -> bytes:
    """Write records incrementally; UTF-8 BOM supports Excel's Unicode detection."""
    with BytesIO() as buffer:
        with TextIOWrapper(buffer, encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=DATASET_COLUMNS)
            writer.writeheader()
            writer.writerows(movie_to_dataset_row(movie) for movie in movies)
            output.flush()
            return buffer.getvalue()

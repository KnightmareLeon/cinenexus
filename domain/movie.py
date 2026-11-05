from dataclasses import dataclass


@dataclass(frozen=True)
class Movie:
    id: str
    title: str
    year: str
    overview: str
    genres: tuple[str, ...]
    director: str | None
    actors: tuple[str, ...]
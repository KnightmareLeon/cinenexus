from dataclasses import dataclass

@dataclass(frozen=True)
class Movie:
    id: str
    title: str
    year: str
    overview: str
    tagline: str
    duration: str
    rating: str
    genres: tuple[str, ...]
    director: str | None
    actors: tuple[str, ...]
    studios: tuple[str, ...]
    themes: tuple[str, ...]
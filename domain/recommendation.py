from dataclasses import dataclass
from math import isfinite

from domain.movie import Movie


@dataclass(frozen=True)
class MovieMatch:
    movie: Movie
    similarity: float

    def __post_init__(self):
        if not isfinite(self.similarity):
            raise ValueError("Similarity score must be finite.")
from pathlib import Path

import pandas as pd

from domain.movie import Movie

class MovieRepository:
    FILE_COLUMNS = {
        "movies.csv": [
            "id", "name", "date", "tagline",
            "description", "minute", "rating"
        ],
        "actors.csv": ["id", "name", "role"],
        "crew.csv": ["id", "role", "name"],
        "genres.csv": ["id", "genre"],
        "studios.csv": ["id", "studio"],
        "themes.csv": ["id", "theme"],
    }

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        df = pd.read_csv(path, dtype={"id": "string"}).fillna("")

        required = self.FILE_COLUMNS[filename]
        missing = [column for column in required if column not in df.columns]

        if missing:
            raise ValueError(f"{filename} is missing required columns: {missing}")

        return df

    @staticmethod
    def _group_values(df: pd.DataFrame, value_column: str, role_filter: str | None = None) -> dict[str, tuple[str, ...]]:
        grouped = {}

        if role_filter is not None:
            df = df[df["role"].astype(str).str.strip().str.casefold() == role_filter.casefold()]

        for movie_id, value in zip(df["id"], df[value_column]):
            movie_id = str(movie_id).strip()
            value = str(value).strip()

            if not movie_id or not value:
                continue

            values = grouped.setdefault(movie_id, [])

            if value not in values:
                values.append(value)

        return {movie_id: tuple(values) for movie_id, values in grouped.items()}

    def load_dataframe(self) -> pd.DataFrame:
        movies = self._read_csv("movies.csv")
        actors = self._read_csv("actors.csv")
        crew = self._read_csv("crew.csv")
        genres = self._read_csv("genres.csv")
        studios = self._read_csv("studios.csv")
        themes = self._read_csv("themes.csv")

        actor_map = self._group_values(actors, "name")
        director_map = self._group_values(crew, "name", role_filter="director")
        genre_map = self._group_values(genres, "genre")
        studio_map = self._group_values(studios, "studio")
        theme_map = self._group_values(themes, "theme")

        df = movies.rename(columns={"name": "title", "date": "year", "description": "overview", "minute": "duration",}).copy()

        df["actors"] = df["id"].map(lambda movie_id: actor_map.get(str(movie_id), ()))
        df["genres"] = df["id"].map(lambda movie_id: genre_map.get(str(movie_id), ()))
        df["studios"] = df["id"].map(lambda movie_id: studio_map.get(str(movie_id), ()))
        df["themes"] = df["id"].map(lambda movie_id: theme_map.get(str(movie_id), ()))
        df["director"] = df["id"].map(lambda movie_id: (director_map.get(str(movie_id), ("",))[0]))

        def build_search_text(row):
            parts = [row["title"], row["tagline"], row["overview"],]

            return ". ".join(str(part).strip() for part in parts if str(part).strip())

        df["search_text"] = df.apply(build_search_text, axis=1)

        return df

    @staticmethod
    def _split_entities(value) -> tuple[str, ...]:
        if value is None:
            return ()

        if isinstance(value, str):
            if not value.strip():
                return ()

            return tuple(item.strip() for item in value.split("|")if item.strip())

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (tuple, list)):
            return tuple(str(item).strip()for item in value if item is not None and str(item).strip())

        return ()

    def get_all(self, df: pd.DataFrame | None = None) -> list[Movie]:
        if df is None:
            df = self.load_dataframe()

        movies = []

        for _, row in df.iterrows():
            director = str(row["director"]).strip()

            movies.append(Movie(
                id=str(row["id"]),
                title=str(row["title"]),
                year=str(row["year"]),
                overview=str(row["overview"]),
                tagline=str(row["tagline"]),
                duration=str(row["duration"]),
                rating=str(row["rating"]),
                genres=self._split_entities(row["genres"]),
                director=director or None,
                actors=self._split_entities(row["actors"]),
                studios=self._split_entities(row["studios"]),
                themes=self._split_entities(row["themes"]),
            ))

        return movies
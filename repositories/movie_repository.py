from pathlib import Path

import pandas as pd

from domain.movie import Movie

class MovieRepository:

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df = df.fillna("")

        required_columns = ["id", "title", "year", "overview", "genres", "director", "actors"]

        missing = [column for column in required_columns if column not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for column in required_columns:
            df[column] = df[column].astype(str)

        df["search_text"] = (df["title"] + ". " + df["overview"] + ". Genres: " + df["genres"])

        return df

    @staticmethod
    def _split_entities(value):
        return tuple(item.strip() for item in str(value).split("|") if item.strip())

    def get_all(self) -> list[Movie]:
        df = self.load_dataframe()
        movies = []

        for _, row in df.iterrows():
            director = row["director"].strip()

            movies.append(
                Movie(
                    id=row["id"],
                    title=row["title"],
                    year=row["year"],
                    overview=row["overview"],
                    genres=self._split_entities(row["genres"]),
                    director=director or None,
                    actors=self._split_entities(row["actors"])
                )
            )

        return movies
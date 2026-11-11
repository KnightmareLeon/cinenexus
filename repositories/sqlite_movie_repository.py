import sqlite3

from collections import defaultdict
from contextlib import closing
from pathlib import Path

from domain.movie import Movie

from repositories.rdbms_movie_repository import RDBMSMovieRepository

class SQLiteMovieRepository(RDBMSMovieRepository):
    RELATIONSHIP_TABLES = {
        "actors": ("movie_actors", "actor_name"),
        "directors": ("movie_directors", "director_name"),
        "genres": ("movie_genres", "genre"),
        "studios": ("movie_studios", "studio"),
        "themes": ("movie_themes", "theme"),
    }

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _load_relationships(self, movie_ids):
        relationships = {
            key: defaultdict(list)
            for key in self.RELATIONSHIP_TABLES
        }

        movie_ids = list(dict.fromkeys(movie_ids))

        if not movie_ids:
            return relationships

        chunk_size = 500

        with closing(self._connect()) as connection:
            for start in range(0, len(movie_ids), chunk_size):
                chunk = movie_ids[start:start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)

                for key, (table, column) in self.RELATIONSHIP_TABLES.items():
                    query = f"""
                        SELECT movie_id, {column}
                        FROM {table}
                        WHERE movie_id IN ({placeholders})
                        ORDER BY movie_id, {column}
                    """

                    rows = connection.execute(
                        query, chunk
                    ).fetchall()

                    for row in rows:
                        relationships[key][row["movie_id"]].append(
                            row[column]
                        )

        return relationships

    @staticmethod
    def _to_movie(row, relationships):
        movie_id = row["id"]

        directors = relationships["directors"].get(movie_id, [])

        return Movie(
            id=movie_id,
            title=row["title"],
            year=row["year"] or "",
            overview=row["overview"] or "",
            tagline=row["tagline"] or "",
            duration=row["duration"] or "",
            rating=row["rating"] or "",
            genres=tuple(relationships["genres"].get(movie_id, [])),
            director=directors[0] if directors else None,
            actors=tuple(relationships["actors"].get(movie_id, [])),
            studios=tuple(relationships["studios"].get(movie_id, [])),
            themes=tuple(relationships["themes"].get(movie_id, [])),
        )

    def _rows_to_movies(self, rows):
        if not rows:
            return []

        movie_ids = [row["id"] for row in rows]
        relationships = self._load_relationships(movie_ids)

        return [self._to_movie(row, relationships) for row in rows]

    def get_movie_by_id(self, movie_id: str) -> Movie | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM movies WHERE id = ?", (str(movie_id),),).fetchone()

        if row is None:
            return None

        return self._rows_to_movies([row])[0]

    def get_movies_by_ids(self, movie_ids: list[str]) -> list[Movie]:
        movie_ids = list(dict.fromkeys(str(mid) for mid in movie_ids))

        if not movie_ids:
            return []

        rows_by_id = {}

        with closing(self._connect()) as connection:
            chunk_size = 500

            for start in range(0, len(movie_ids), chunk_size):
                chunk = movie_ids[start:start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)

                rows = connection.execute(
                    f"""
                    SELECT *
                    FROM movies
                    WHERE id IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()

                rows_by_id.update({row["id"]: row for row in rows})

        ordered_rows = [rows_by_id[movie_id] for movie_id in movie_ids if movie_id in rows_by_id]

        return self._rows_to_movies(ordered_rows)

    def get_movies_page(self, page: int = 1, page_size: int = 50) -> list[Movie]:
        if page < 1:
            raise ValueError("Page must be at least 1.")

        if page_size < 1:
            raise ValueError("Page size must be at least 1.")

        offset = (page - 1) * page_size

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM movies
                ORDER BY normalized_title, id
                LIMIT ? OFFSET ?
                """,
                (page_size, offset),
            ).fetchall()

        return self._rows_to_movies(rows)

    def iter_movies(self, batch_size: int = 500):
        """Stream the catalog and hydrate relationships in batches."""
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1.")

        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "SELECT * FROM movies ORDER BY normalized_title, id"
            )
            while rows := cursor.fetchmany(batch_size):
                yield from self._rows_to_movies(rows)

    def count_movies(self) -> int:
        with closing(self._connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM movies").fetchone()[0]

    def search_by_title(self, query: str, limit: int = 20) -> list[Movie]:
        query = query.strip().casefold()

        if not query:
            return []

        if limit < 1:
            raise ValueError("Limit must be at least 1.")

        escaped_query = (query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM movies
                WHERE normalized_title LIKE ? ESCAPE '\\'
                ORDER BY normalized_title, id
                LIMIT ?
                """,
                (escaped_query + "%", limit),
            ).fetchall()

        return self._rows_to_movies(rows)

    def get_related_movies(self, movie_id: str, limit: int = 20):
        query = """
            WITH focus_actors AS (
                SELECT actor_name
                FROM movie_actors
                WHERE movie_id = ?
            ),
            focus_directors AS (
                SELECT director_name
                FROM movie_directors
                WHERE movie_id = ?
            ),
            focus_genres AS (
                SELECT genre
                FROM movie_genres
                WHERE movie_id = ?
            ),
            candidate_matches AS (
                -- Shared actors
                SELECT
                    candidate.movie_id,
                    COUNT(*) AS shared_count
                FROM focus_actors AS focus
                JOIN movie_actors AS candidate
                    INDEXED BY idx_actors_name
                    ON candidate.actor_name = focus.actor_name
                WHERE candidate.movie_id != ?
                GROUP BY candidate.movie_id

                UNION ALL

                -- Shared directors
                SELECT
                    candidate.movie_id,
                    COUNT(*) AS shared_count
                FROM focus_directors AS focus
                JOIN movie_directors AS candidate
                    INDEXED BY idx_directors_name
                    ON candidate.director_name = focus.director_name
                WHERE candidate.movie_id != ?
                GROUP BY candidate.movie_id

                UNION ALL

                -- Shared genres
                SELECT
                    candidate.movie_id,
                    COUNT(*) AS shared_count
                FROM focus_genres AS focus
                JOIN movie_genres AS candidate
                    INDEXED BY idx_genres_name
                    ON candidate.genre = focus.genre
                WHERE candidate.movie_id != ?
                GROUP BY candidate.movie_id
            ),
            ranked_movies AS (
                SELECT
                    movie_id,
                    SUM(shared_count) AS total_shared
                FROM candidate_matches
                GROUP BY movie_id
            )
            SELECT movie_id
            FROM ranked_movies
            ORDER BY total_shared DESC, movie_id
            LIMIT ?
        """

        with self._connect() as connection:
            rows = connection.execute(
                query,
                (
                    movie_id,
                    movie_id,
                    movie_id,
                    movie_id,
                    movie_id,
                    movie_id,
                    limit,
                ),
            ).fetchall()

        movie_ids = [row["movie_id"] for row in rows]

        if not movie_ids:
            return []

        movies = self.get_movies_by_ids(movie_ids)
        movies_by_id = {movie.id: movie for movie in movies}

        return [
            movies_by_id[related_id]
            for related_id in movie_ids
            if related_id in movies_by_id
        ]

    def get_movie_embedding_index(self, movie_id: str) -> int | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT embedding_index
                FROM movies
                WHERE id = ?
                """,
                (str(movie_id),),
            ).fetchone()

        return int(row["embedding_index"]) if row else None

    def get_movie_ids_by_embedding_indices(self, embedding_indices: list[int]) -> list[str]:
        embedding_indices = [int(index) for index in embedding_indices]

        if not embedding_indices:
            return []

        ids_by_index = {}

        with closing(self._connect()) as connection:
            chunk_size = 500

            for start in range(0, len(embedding_indices), chunk_size):
                chunk = embedding_indices[start:start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)

                rows = connection.execute(
                    f"""
                    SELECT id, embedding_index
                    FROM movies
                    WHERE embedding_index IN ({placeholders})
                    """,
                    chunk,
                ).fetchall()

                for row in rows:
                    ids_by_index[int(row["embedding_index"])] = row["id"]

        return [ids_by_index[index] for index in embedding_indices if index in ids_by_index]

    
    def get_embedding_metadata(self) -> dict[str, str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT key, value
                FROM embedding_metadata
                """
            ).fetchall()

        return {row["key"]: row["value"] for row in rows}

    def get_search_texts(self) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT search_text
                FROM movies
                ORDER BY embedding_index
                """
            ).fetchall()

        return [row["search_text"] for row in rows]

    def iter_search_texts(self):
        """Stream search texts in embedding order for fingerprinting and TF-IDF."""
        with closing(self._connect()) as connection:
            cursor = connection.execute("SELECT search_text FROM movies ORDER BY embedding_index")
            for row in cursor:
                yield row["search_text"]

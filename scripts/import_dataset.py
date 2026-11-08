from pathlib import Path

from repositories.database_repository import DatabaseRepository
from repositories.movie_repository import MovieRepository

from ml.model import MODEL_NAME
from repositories.embedding_repository import EmbeddingRepository

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
DATABASE_PATH = DATA_DIR / "cinenexus.db"

BATCH_SIZE = 5000

def insert_batch(connection, movie_rows, actor_rows, director_rows, genre_rows, studio_rows, theme_rows,):
    if movie_rows:
        connection.executemany(
            """
            INSERT INTO movies (
                id, title, normalized_title, year,
                tagline, overview, duration, rating,
                search_text, embedding_index
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            movie_rows,
        )

    relationship_tables = [
        ("movie_actors", "movie_id, actor_name", actor_rows),
        ("movie_directors", "movie_id, director_name", director_rows),
        ("movie_genres", "movie_id, genre", genre_rows),
        ("movie_studios", "movie_id, studio", studio_rows),
        ("movie_themes", "movie_id, theme", theme_rows),
    ]

    for table, columns, rows in relationship_tables:
        if rows:
            connection.executemany(
                f"INSERT INTO {table} ({columns}) VALUES (?, ?)",
                rows,
            )


def import_dataset():
    print("Loading CSV files...")

    movie_repository = MovieRepository(DATA_DIR)
    dataframe = movie_repository.load_dataframe()

    total_movies = len(dataframe)
    print(f"Loaded {total_movies:,} movie records.")

    database = DatabaseRepository(DATABASE_PATH)
    database.initialize()

    search_texts = dataframe["search_text"].tolist()

    embedding_fingerprint = EmbeddingRepository.create_fingerprint(search_texts, MODEL_NAME)
    connection = database.connect()

    try:
        with connection:
            print("Clearing existing database records...")

            connection.execute("DELETE FROM movies")
            connection.execute("DELETE FROM embedding_metadata")

            connection.executemany(
                """
                INSERT INTO embedding_metadata (key, value)
                VALUES (?, ?)
                """,
                [
                    ("fingerprint", embedding_fingerprint),
                    ("model_name", MODEL_NAME),
                    ("embedding_count", str(total_movies)),
                ],
            )

            movie_rows = []
            actor_rows = []
            director_rows = []
            genre_rows = []
            studio_rows = []
            theme_rows = []

            print("Importing movie records...")

            for embedding_index, row in enumerate(dataframe.itertuples(index=False)):
                movie_id = str(row.id).strip()

                movie_rows.append((
                    movie_id,
                    str(row.title),
                    str(row.title).strip().casefold(),
                    str(row.year),
                    str(row.tagline),
                    str(row.overview),
                    str(row.duration),
                    str(row.rating),
                    str(row.search_text),
                    embedding_index,
                ))

                for actor in movie_repository._split_entities(row.actors):
                    actor_rows.append((movie_id, actor))

                director = str(row.director).strip()
                if director:
                    director_rows.append((movie_id, director))

                for genre in movie_repository._split_entities(row.genres):
                    genre_rows.append((movie_id, genre))

                for studio in movie_repository._split_entities(row.studios):
                    studio_rows.append((movie_id, studio))

                for theme in movie_repository._split_entities(row.themes):
                    theme_rows.append((movie_id, theme))

                if len(movie_rows) >= BATCH_SIZE:
                    insert_batch(connection, movie_rows, actor_rows, director_rows, genre_rows, studio_rows, theme_rows,)

                    movie_rows.clear()
                    actor_rows.clear()
                    director_rows.clear()
                    genre_rows.clear()
                    studio_rows.clear()
                    theme_rows.clear()
                    
                    print(
                        f"Imported {embedding_index + 1:,} "
                        f"of {total_movies:,} movies..."
                    )

            insert_batch(connection, movie_rows, actor_rows, director_rows, genre_rows, studio_rows, theme_rows,)

            print("Importing completed successfully.")

        with connection:
            imported_count = connection.execute(
                "SELECT COUNT(*) FROM movies"
            ).fetchone()[0]

        print(f"Total movies in database: {imported_count:,}")
        print(f"Database location: {DATABASE_PATH}")

    finally:
        connection.close()


if __name__ == "__main__":
    import_dataset()

import sqlite3
from pathlib import Path

class DatabaseRepository:
    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)

    def connect(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")

        return connection

    def initialize(self):
        with self.connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS movies (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    normalized_title TEXT NOT NULL,
                    year TEXT,
                    tagline TEXT,
                    overview TEXT,
                    duration TEXT,
                    rating TEXT,
                    search_text TEXT,
                    embedding_index INTEGER UNIQUE NOT NULL
                );

                CREATE TABLE IF NOT EXISTS movie_actors (
                    movie_id TEXT NOT NULL,
                    actor_name TEXT NOT NULL,

                    PRIMARY KEY (movie_id, actor_name),

                    FOREIGN KEY (movie_id)
                        REFERENCES movies(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS movie_directors (
                    movie_id TEXT NOT NULL,
                    director_name TEXT NOT NULL,

                    PRIMARY KEY (movie_id, director_name),

                    FOREIGN KEY (movie_id)
                        REFERENCES movies(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS movie_genres (
                    movie_id TEXT NOT NULL,
                    genre TEXT NOT NULL,

                    PRIMARY KEY (movie_id, genre),

                    FOREIGN KEY (movie_id)
                        REFERENCES movies(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS movie_studios (
                    movie_id TEXT NOT NULL,
                    studio TEXT NOT NULL,

                    PRIMARY KEY (movie_id, studio),

                    FOREIGN KEY (movie_id)
                        REFERENCES movies(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS movie_themes (
                    movie_id TEXT NOT NULL,
                    theme TEXT NOT NULL,

                    PRIMARY KEY (movie_id, theme),

                    FOREIGN KEY (movie_id)
                        REFERENCES movies(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embedding_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_movies_title
                    ON movies(normalized_title);

                CREATE INDEX IF NOT EXISTS idx_actors_name
                    ON movie_actors(actor_name, movie_id);

                CREATE INDEX IF NOT EXISTS idx_directors_name
                    ON movie_directors(director_name, movie_id);

                CREATE INDEX IF NOT EXISTS idx_genres_name
                    ON movie_genres(genre, movie_id);

                CREATE INDEX IF NOT EXISTS idx_studios_name
                    ON movie_studios(studio, movie_id);

                CREATE INDEX IF NOT EXISTS idx_themes_name
                    ON movie_themes(theme, movie_id);
            """)

    def get_connection(self):
        return self.connect()

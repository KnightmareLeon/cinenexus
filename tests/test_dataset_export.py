import csv
import io
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from repositories.database_repository import DatabaseRepository
from repositories.sqlite_movie_repository import SQLiteMovieRepository
from services.dataset_export_service import DATASET_COLUMNS, movies_to_csv
from services.movie_service import MovieService
from services.query_cache import get_cached_catalog_csv, get_cached_movies_page


def create_catalog(database_path, count=53):
    database = DatabaseRepository(database_path)
    with closing(database.connect()) as connection:
        with patch.object(database, "connect", return_value=connection):
            database.initialize()
    with closing(database.connect()) as connection, connection:
        connection.executemany(
            """INSERT INTO movies
            (id, title, normalized_title, year, overview, embedding_index)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [(f"id-{i}", f'Movie {i:03d}, "café"', f"movie {i:03d}", "2024",
              'A quoted "story", with commas.\nAnother line: 日本語.', i)
             for i in range(count)],
        )
        if count:
            connection.executemany(
                "INSERT INTO movie_genres VALUES (?, ?)",
                [("id-0", "Drama"), ("id-0", "Comedy")],
            )
            connection.execute("INSERT INTO movie_actors VALUES ('id-0', 'Zoë Actor')")
            connection.execute("INSERT INTO movie_directors VALUES ('id-0', 'René Director')")
    return MovieService(SQLiteMovieRepository(database_path))


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.path = Path(self.temp_directory.name) / "catalog.db"
        self.service = create_catalog(self.path)
        get_cached_catalog_csv.clear()
        get_cached_movies_page.clear()

    def test_full_export_batches_all_records_and_round_trips_csv(self):
        repository = self.service.movie_repository
        with patch.object(repository, "_load_relationships", wraps=repository._load_relationships) as hydrate:
            data = movies_to_csv(self.service.iter_movies(batch_size=7))
        rows = read_csv(data)
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(tuple(rows[0]), DATASET_COLUMNS)
        self.assertEqual(len(rows), 53)
        self.assertEqual(len({row["Title"] for row in rows}), 53)
        self.assertEqual(rows[-1]["Title"], 'Movie 052, "café"')
        self.assertEqual(rows[0]["Genres"], "Comedy, Drama")
        self.assertEqual(rows[0]["Director"], "René Director")
        self.assertEqual(rows[0]["Actors"], "Zoë Actor")
        self.assertEqual(rows[0]["Overview"], 'A quoted "story", with commas.\nAnother line: 日本語.')
        self.assertEqual(rows[1]["Director"], "Unknown")
        self.assertEqual([len(call.args[0]) for call in hydrate.call_args_list], [7] * 7 + [4])

    def test_export_cache_reused_and_invalidated_by_fingerprint(self):
        with patch.object(self.service, "iter_movies", wraps=self.service.iter_movies) as iterate:
            first = get_cached_catalog_csv(self.service, "dataset-one")
            self.assertEqual(first, get_cached_catalog_csv(self.service, "dataset-one"))
            self.assertEqual(iterate.call_count, 1)
            get_cached_catalog_csv(self.service, "dataset-two")
            self.assertEqual(iterate.call_count, 2)

    def test_empty_export_has_header_and_invalid_batch_size_is_rejected(self):
        self.assertEqual(movies_to_csv([]).decode("utf-8-sig"), ",".join(DATASET_COLUMNS) + "\r\n")
        with self.assertRaises(ValueError):
            list(self.service.iter_movies(batch_size=0))

    def dataset_app(self):
        app = AppTest.from_string('''
from pathlib import Path
import streamlit as st
from repositories.sqlite_movie_repository import SQLiteMovieRepository
from services.movie_service import MovieService
from presentation.dataset_page import render_dataset_page
service = MovieService(SQLiteMovieRepository(Path(st.session_state.test_database)))
render_dataset_page(service, st.session_state.get("test_fingerprint", "dataset-ui-one"))
''')
        app.session_state["test_database"] = str(self.path)
        return app

    def test_page_export_and_full_export_are_lazy_and_persist_across_navigation(self):
        with patch.object(SQLiteMovieRepository, "iter_movies", autospec=True,
                          side_effect=SQLiteMovieRepository.iter_movies) as iterate:
            app = self.dataset_app().run()
            self.assertFalse(app.exception)
            self.assertEqual(iterate.call_count, 0)
            self.assertEqual(len(app.dataframe[0].value), 50)
            self.assertEqual([item.label for item in app.get("download_button")], ["Download current page (CSV)"])
            app.button(key="dataset_prepare_csv").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(iterate.call_count, 1)
            self.assertEqual(len(app.get("download_button")), 2)
            app.button(key="dataset_next").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value), 3)
            self.assertEqual(iterate.call_count, 1)
            self.assertEqual(app.session_state.dataset_export_fingerprint, "dataset-ui-one")
            app.session_state["test_fingerprint"] = "dataset-ui-two"
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("download_button")), 1)

    def test_empty_catalog_has_no_download(self):
        with closing(self.service.movie_repository._connect()) as connection, connection:
            connection.execute("DELETE FROM movies")
        app = self.dataset_app().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("download_button")), 0)
        self.assertTrue(any("No movie records" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()

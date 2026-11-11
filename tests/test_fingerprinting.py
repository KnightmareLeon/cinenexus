import hashlib
import sqlite3
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from repositories.embedding_repository import EmbeddingRepository
from repositories.sqlite_movie_repository import SQLiteMovieRepository
from services.embedding_service import get_movie_embeddings_from_database


class FingerprintingTests(unittest.TestCase):
    def test_one_shot_iterator_preserves_existing_digest_format(self):
        texts = ["First movie", "", "日本語 café", None, 42]
        expected_bytes = b"test-model\0First movie\0\0" + "日本語 café".encode("utf-8") + b"\0None\0" + b"42"
        expected = hashlib.sha256(expected_bytes).hexdigest()
        self.assertEqual(EmbeddingRepository.create_fingerprint(iter(texts), "test-model"), expected)
        self.assertEqual(EmbeddingRepository.create_fingerprint([], "test-model"),
                         hashlib.sha256(b"test-model").hexdigest())
        self.assertNotEqual(EmbeddingRepository.create_fingerprint(reversed(texts), "test-model"), expected)
        self.assertNotEqual(EmbeddingRepository.create_fingerprint(iter(texts), "other-model"), expected)

    def test_fingerprinting_does_not_retain_generated_corpus(self):
        payload = "x" * 4096
        texts = (f"{index}:{payload}" for index in range(5000))
        tracemalloc.start()
        try:
            EmbeddingRepository.create_fingerprint(texts, "test-model")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Materializing this corpus would retain over 20 MB of unique strings.
        self.assertLess(peak, 1_000_000)

    def test_sqlite_fingerprint_uses_embedding_order(self):
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE movies (id TEXT, search_text TEXT, embedding_index INTEGER)")
        connection.executemany("INSERT INTO movies VALUES (?, ?, ?)",
                               [("a", "third", 2), ("z", "first", 0), ("m", "second", 1)])
        repository = SQLiteMovieRepository(Path("unused.db"))
        with patch.object(repository, "_connect", return_value=connection):
            actual = EmbeddingRepository.create_fingerprint(repository.iter_search_texts(), "test-model")
        expected = hashlib.sha256(b"test-model\0first\0second\0third").hexdigest()
        self.assertEqual(actual, expected)

    def test_cached_embeddings_skip_database_reads(self):
        service = Mock()
        repository = Mock(spec=EmbeddingRepository)
        embeddings = np.array([[1., 0.]], dtype=np.float32)
        repository.load.return_value = embeddings
        actual = get_movie_embeddings_from_database(service, Mock(), "test-model", repository, "saved")
        self.assertIs(actual, embeddings)
        service.iter_search_texts.assert_not_called()
        service.get_search_texts.assert_not_called()

    def test_mismatch_rejected_before_loading_or_encoding_full_corpus(self):
        service = Mock()
        service.iter_search_texts.return_value = iter(["changed text"])
        repository = Mock(spec=EmbeddingRepository)
        repository.load.return_value = None
        repository.create_fingerprint.side_effect = EmbeddingRepository.create_fingerprint
        with patch("services.embedding_service.encode_texts") as encode:
            with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
                get_movie_embeddings_from_database(service, Mock(), "test-model", repository, "old")
        service.get_search_texts.assert_not_called()
        encode.assert_not_called()
        repository.save.assert_not_called()

    def test_matching_stream_validated_before_embedding_generation(self):
        events = []
        service = Mock()

        def stream():
            for text in ["first", "second"]:
                events.append("hash")
                yield text

        def load_texts():
            events.append("load corpus")
            return ["first", "second"]

        service.iter_search_texts.return_value = stream()
        service.get_search_texts.side_effect = load_texts
        repository = Mock(spec=EmbeddingRepository)
        repository.load.return_value = None
        repository.create_fingerprint.side_effect = EmbeddingRepository.create_fingerprint
        expected = hashlib.sha256(b"test-model\0first\0second").hexdigest()
        model = Mock()
        embeddings = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        with patch("services.embedding_service.encode_texts", return_value=embeddings) as encode:
            actual = get_movie_embeddings_from_database(service, model, "test-model", repository, expected)
        self.assertEqual(events, ["hash", "hash", "load corpus"])
        encode.assert_called_once_with(["first", "second"], model)
        repository.save.assert_called_once_with(embeddings, expected)
        self.assertIs(actual, embeddings)

    def test_importer_hashes_column_without_creating_a_list(self):
        from scripts.import_dataset import import_dataset, MODEL_NAME

        dataframe = pd.DataFrame([
            dict(id="id-b", title="First", year="2024", tagline="", overview="",
                 duration="", rating="", search_text="First", director="", actors=(),
                 genres=(), studios=(), themes=()),
            dict(id="id-a", title="Second", year="2025", tagline="", overview="",
                 duration="", rating="", search_text="Second", director="", actors=(),
                 genres=(), studios=(), themes=()),
        ])
        source = Mock()
        source.load_dataframe.return_value = dataframe
        source._split_entities.return_value = ()
        database = Mock()
        connection = MagicMock()
        database.connect.return_value = connection
        connection.execute.return_value.fetchone.return_value = (2,)
        with patch("scripts.import_dataset.MovieRepository", return_value=source), \
             patch("scripts.import_dataset.DatabaseRepository", return_value=database), \
             patch.object(pd.Series, "tolist", side_effect=AssertionError("Do not copy the corpus")), \
             patch("builtins.print"):
            import_dataset()
        expected = EmbeddingRepository.create_fingerprint(iter(["First", "Second"]), MODEL_NAME)
        metadata = connection.executemany.call_args_list[0].args[1]
        self.assertIn(("fingerprint", expected), metadata)
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import hashlib

import numpy as np


class EmbeddingRepository:

    def __init__(self, file_path: Path):
        self.file_path = file_path

    @staticmethod
    def create_fingerprint(texts, model_name):
        hasher = hashlib.sha256()

        hasher.update(model_name.encode("utf-8"))

        for text in texts:
            hasher.update(b"\0")
            hasher.update(str(text).encode("utf-8"))

        return hasher.hexdigest()

    def load(self, expected_fingerprint):
        if not self.file_path.exists():
            return None

        with np.load(self.file_path, allow_pickle=False) as data:
            fingerprint = str(data["fingerprint"].item())

            if fingerprint != expected_fingerprint:
                return None

            return data["embeddings"].copy()

    def save(self, embeddings, fingerprint):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            self.file_path,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            fingerprint=np.array(fingerprint)
        )
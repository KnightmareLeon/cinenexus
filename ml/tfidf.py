import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfIndex:
    """Sparse, normalized movie vectors in database embedding-index order."""

    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=100_000, dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        if top_k <= 0 or not query.strip():
            return [], []
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return [], []
        # Both vectors are L2-normalized, so their dot product is cosine similarity.
        scores = (self.matrix @ query_vector.T).tocoo()
        positive = scores.data > 0
        indices, values = scores.row[positive], scores.data[positive]
        order = np.lexsort((indices, -values))[:top_k]
        return indices[order].tolist(), values[order].tolist()

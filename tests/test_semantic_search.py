import json
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from app.core.config import SearchConfig
from app.retrieval.semantic_search import SemanticSearchEngine

RECORDS = [
    {"id": "P101", "name": "Chicken Burger", "description": "Grilled chicken with cheese", "category": "Burgers"},
    {"id": "P102", "name": "Vegetable Pizza", "description": "Pizza with fresh vegetables", "category": "Pizza"},
    {"id": "P104", "name": "Cheese Pizza", "category": "Pizza"},
    {"id": "P105", "price": 3.0},
]


class BagOfWordsEmbedder:
    """Deterministic stand-in for the sentence-transformer model."""

    dimension = 64

    def __init__(self):
        self.document_calls = 0

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.encode("utf-8")) % self.dimension] += 1.0
        return vector / max(np.linalg.norm(vector), 1e-12)

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.stack([self._encode(text) for text in texts])

    def encode_query(self, query):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        return self._encode(query).reshape(1, -1)


class SemanticSearchEngineTests(unittest.TestCase):

    def setUp(self):
        self.engine = SemanticSearchEngine(SearchConfig(), embedder=BagOfWordsEmbedder())

    def test_skips_records_without_searchable_text(self):
        self.engine.build(RECORDS)

        self.assertEqual(self.engine.total_documents, 3)
        self.assertNotIn("P105", self.engine.records)

    def test_search_ranks_matching_records_first(self):
        self.engine.build(RECORDS)

        results = self.engine.search("burger", top_k=3)

        self.assertEqual(results[0]["id"], "P101")
        self.assertEqual(results[0]["item"]["name"], "Chicken Burger")

    def test_search_validates_top_k_and_query(self):
        self.engine.build(RECORDS)

        with self.assertRaises(ValueError):
            self.engine.search("pizza", top_k=1000)

        with self.assertRaises(ValueError):
            self.engine.search("   ")

    def test_search_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.engine.search("pizza")

    def test_load_or_build_reuses_saved_index_until_dataset_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "products.json"
            storage = Path(directory) / "storage"
            dataset.write_text(json.dumps(RECORDS), encoding="utf-8")

            self.assertTrue(self.engine.load_or_build(dataset, storage))

            embedder = BagOfWordsEmbedder()
            reloaded = SemanticSearchEngine(SearchConfig(), embedder=embedder)
            self.assertFalse(reloaded.load_or_build(dataset, storage))
            self.assertEqual(embedder.document_calls, 0)
            self.assertEqual(reloaded.search("burger")[0]["id"], "P101")

            dataset.write_text(json.dumps(RECORDS[:2]), encoding="utf-8")
            self.assertTrue(reloaded.load_or_build(dataset, storage))
            self.assertEqual(reloaded.total_documents, 2)


if __name__ == "__main__":
    unittest.main()

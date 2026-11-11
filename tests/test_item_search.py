import unittest
import zlib

import numpy as np

from app.core.config import SearchConfig
from app.retrieval.item_search import ItemSearcher

MEALS = [
    {"id": 1, "name": "Classic Cheeseburger", "description": "Beef patty, cheddar, pickles", "category": "Burgers"},
    {"id": 2, "name": "Margherita Pizza", "description": "Tomato, mozzarella, basil", "category": "Pizza"},
    {"id": 3, "name": "Chicken Shawarma", "description": "Chicken wrap with garlic sauce", "category": "Wraps",
     "ingredients": ["chicken", "garlic", "pickles"]},
    {"id": 4, "price": 2.0},
]


class BagOfWordsEmbedder:
    dimension = 64

    def __init__(self):
        self.encoded_documents = 0

    def _encode(self, text):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in text.lower().replace(",", " ").split():
            vector[zlib.crc32(word.encode("utf-8")) % self.dimension] += 1.0
        return vector / max(np.linalg.norm(vector), 1e-12)

    def encode_documents(self, texts):
        self.encoded_documents += len(texts)
        return np.stack([self._encode(text) for text in texts])

    def encode_query(self, query):
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        return self._encode(query).reshape(1, -1)


class ItemSearcherTests(unittest.TestCase):

    def setUp(self):
        self.embedder = BagOfWordsEmbedder()
        self.searcher = ItemSearcher(SearchConfig().collections, self.embedder)

    def test_ranks_items_and_returns_original_objects(self):
        results = self.searcher.search("delivery-meals", "chicken garlic", MEALS)

        self.assertEqual(results[0]["id"], "3")
        self.assertIs(results[0]["item"], MEALS[2])
        self.assertEqual(len(results), 3)

    def test_uses_collection_fields(self):
        products = [
            {"id": 1, "title": "Phone X Pro", "description": "Fast phone", "brand": "Apex"},
            {"id": 2, "title": "Laptop Air", "description": "Light laptop", "brand": "Nova"},
        ]

        results = self.searcher.search("store-products", "laptop", products)

        self.assertEqual(results[0]["id"], "2")

    def test_top_k_and_min_score(self):
        self.assertEqual(
            len(self.searcher.search("delivery-meals", "pizza", MEALS, top_k=1)), 1
        )

        results = self.searcher.search("delivery-meals", "pizza", MEALS, min_score=0.1)
        self.assertEqual([result["id"] for result in results], ["2"])

    def test_reuses_cached_embeddings(self):
        self.searcher.search("delivery-meals", "burger", MEALS)
        encoded = self.embedder.encoded_documents

        self.searcher.search("delivery-meals", "pizza", MEALS)

        self.assertEqual(self.embedder.encoded_documents, encoded)

    def test_cache_size_is_bounded(self):
        searcher = ItemSearcher(SearchConfig().collections, self.embedder, cache_size=2)

        searcher.search("delivery-meals", "burger", MEALS)

        self.assertEqual(len(searcher._cache), 2)

    def test_empty_items_and_unknown_collection(self):
        self.assertEqual(self.searcher.search("delivery-meals", "burger", []), [])

        with self.assertRaises(KeyError):
            self.searcher.search("unknown", "burger", MEALS)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.searcher.search("delivery-meals", "   ", MEALS)

        with self.assertRaises(ValueError):
            self.searcher.search("delivery-meals", "burger", [{"name": "No ID"}])


if __name__ == "__main__":
    unittest.main()

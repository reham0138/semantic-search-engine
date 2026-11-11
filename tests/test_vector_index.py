import tempfile
import unittest

import numpy as np

from app.indexing.vector_index import VectorIndex


def build_index() -> VectorIndex:
    index = VectorIndex(dimension=3)
    index.add(
        np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        ["product_1", "product_2", "product_3"]
    )
    return index


class VectorIndexTests(unittest.TestCase):

    def test_search_returns_most_similar_first(self):
        results = build_index().search(np.array([1.0, 0.0, 0.0]), top_k=2)

        self.assertEqual([r["document_id"] for r in results], ["product_1", "product_2"])
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)

    def test_rejects_duplicate_ids(self):
        index = build_index()

        with self.assertRaises(ValueError):
            index.add(np.array([[0.0, 0.0, 1.0]]), ["product_1"])

    def test_save_and_load_round_trip(self):
        index = build_index()

        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            loaded = VectorIndex.load(directory)

        self.assertEqual(loaded.total_vectors, 3)
        self.assertEqual(loaded.document_ids, index.document_ids)
        self.assertEqual(
            loaded.search(np.array([0.0, 1.0, 0.0]), top_k=1)[0]["document_id"],
            "product_3"
        )


if __name__ == "__main__":
    unittest.main()

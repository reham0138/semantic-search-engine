
import json
from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "index.faiss"
IDS_FILE = "document_ids.json"


class VectorIndex:

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.document_ids = []

    def add(
        self,
        embeddings: np.ndarray,
        document_ids: list[str]
    ) -> None:
        """Add document embeddings to the index."""

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if len(embeddings) != len(document_ids):
            raise ValueError(
                "Number of embeddings must match number of document IDs."
            )

        if len(embeddings) == 0:
            raise ValueError("Cannot add an empty embedding list.")

        if any(not doc_id for doc_id in document_ids):
            raise ValueError("Document IDs cannot be empty.")

        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Document IDs must be unique.")

        if set(document_ids).intersection(self.document_ids):
            raise ValueError("Some document IDs already exist in the index.")

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.document_ids.extend(document_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[dict]:
        """Retrieve the most similar documents."""

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.dimension):
            raise ValueError("Invalid query embedding shape.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "document_id": self.document_ids[idx],
                "score": float(score)
            })

        return results

    def save(self, directory: str | Path) -> None:
        """Write the index and its document IDs to a directory."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / INDEX_FILE))

        (directory / IDS_FILE).write_text(
            json.dumps(self.document_ids, ensure_ascii=False),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        """Read an index previously written with save()."""

        directory = Path(directory)

        index = faiss.read_index(str(directory / INDEX_FILE))
        document_ids = json.loads(
            (directory / IDS_FILE).read_text(encoding="utf-8")
        )

        if index.ntotal != len(document_ids):
            raise ValueError("Index and document IDs are out of sync.")

        vector_index = cls(index.d)
        vector_index.index = index
        vector_index.document_ids = document_ids

        return vector_index

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

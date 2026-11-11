
import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        query_prefix: str = "query: ",
        document_prefix: str = "passage: "
    ):
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        self.dimension = self.model.get_embedding_dimension()

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """Generate normalized embeddings for documents."""

        if not texts:
            raise ValueError("Document list cannot be empty.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All documents must be non-empty strings.")

        prepared_texts = [
            f"{self.document_prefix}{text.strip()}"
            for text in texts
        ]

        embeddings = self.model.encode(
            prepared_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Generate a normalized embedding for a search query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [f"{self.query_prefix}{query.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

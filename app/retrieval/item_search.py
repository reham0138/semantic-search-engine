
from collections import OrderedDict

import numpy as np

from app.core.config import CollectionConfig
from app.data.preprocessors import prepare_document


class ItemSearcher:
    """Rank records sent by an application in the request.

    Field embeddings are cached by text, so repeated searches over the same
    catalogue only embed the query.
    """

    def __init__(
        self,
        collections: dict[str, CollectionConfig],
        embedder,
        cache_size: int = 50000
    ):
        self.collections = collections
        self.embedder = embedder
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def search(
        self,
        collection: str,
        query: str,
        items: list[dict],
        top_k: int | None = None,
        min_score: float | None = None,
        max_score_gap: float | None = None
    ) -> list[dict]:
        """Return items scoring at least min_score and within max_score_gap
        of the best match. Unset limits use the collection defaults."""

        if collection not in self.collections:
            raise KeyError(collection)

        config = self.collections[collection]

        if min_score is None:
            min_score = config.min_score

        if max_score_gap is None:
            max_score_gap = config.max_score_gap

        documents = []
        searchable_items = []

        for item in items:
            document = prepare_document(item, config)

            if document["text"]:
                documents.append(document)
                searchable_items.append(item)

        query_vector = self.embedder.encode_query(query).reshape(-1)

        if not documents:
            return []

        vectors = self._embed_documents(documents, config)
        self._evict()

        norms = np.maximum(np.linalg.norm(vectors, axis=1), 1e-12)
        scores = (vectors @ query_vector) / norms

        order = np.argsort(-scores, kind="stable")

        threshold = -np.inf if min_score is None else min_score

        if max_score_gap is not None:
            threshold = max(threshold, float(scores[order[0]]) - max_score_gap)

        results = []

        for i in order:
            score = float(scores[i])

            if score < threshold:
                break

            results.append({
                "id": documents[i]["id"],
                "score": round(score, 4),
                "item": searchable_items[i]
            })

            if top_k is not None and len(results) >= top_k:
                break

        return results

    def _embed_documents(
        self,
        documents: list[dict],
        config: CollectionConfig
    ) -> np.ndarray:
        vectors = np.zeros(
            (len(documents), self.embedder.dimension),
            dtype=np.float32
        )

        for field_name, weight in config.searchable_fields.items():
            rows = [
                (i, document["fields"][field_name])
                for i, document in enumerate(documents)
                if document["fields"].get(field_name)
            ]

            if not rows:
                continue

            embeddings = self._get_embeddings([text for _, text in rows])

            for (i, _), embedding in zip(rows, embeddings):
                vectors[i] += weight * embedding

        return vectors

    def _get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        missing = list(dict.fromkeys(
            text for text in texts if text not in self._cache
        ))

        if missing:
            for text, embedding in zip(
                missing, self.embedder.encode_documents(missing)
            ):
                self._cache[text] = embedding

        for text in texts:
            self._cache.move_to_end(text)

        return [self._cache[text] for text in texts]

    def _evict(self) -> None:
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

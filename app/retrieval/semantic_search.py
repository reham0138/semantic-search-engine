
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from app.core.config import SearchConfig
from app.data.loaders import load_records
from app.data.preprocessors import prepare_document
from app.indexing.vector_index import VectorIndex

logger = logging.getLogger(__name__)

RECORDS_FILE = "records.json"
METADATA_FILE = "metadata.json"


class SemanticSearchEngine:

    def __init__(self, config: SearchConfig | None = None, embedder=None):
        self.config = config or SearchConfig()

        if embedder is None:
            from app.embeddings.model import EmbeddingService

            embedder = EmbeddingService(
                model_name=self.config.model_name,
                query_prefix=self.config.query_prefix,
                document_prefix=self.config.document_prefix
            )

        self.embedder = embedder
        self.index: VectorIndex | None = None
        self.records: dict[str, dict] = {}

    @property
    def total_documents(self) -> int:
        return self.index.total_vectors if self.index else 0

    def build(self, records: list[dict]) -> None:
        """Embed and index a list of application records."""

        documents = []
        records_by_id = {}

        for record in records:
            document = prepare_document(record, self.config)

            if not document["text"]:
                logger.warning(
                    "Skipping record %s: no searchable text.", document["id"]
                )
                continue

            if document["id"] in records_by_id:
                raise ValueError(f"Duplicate ID: {document['id']}")

            documents.append(document)
            records_by_id[document["id"]] = record

        if not documents:
            raise ValueError("No searchable records found.")

        index = VectorIndex(self.embedder.dimension)
        index.add(
            self._embed_documents(documents),
            [document["id"] for document in documents]
        )

        self.index = index
        self.records = records_by_id

    def _embed_documents(self, documents: list[dict]) -> np.ndarray:
        """Combine per-field embeddings using the configured field weights."""

        vectors = np.zeros(
            (len(documents), self.embedder.dimension),
            dtype=np.float32
        )

        for field_name, weight in self.config.searchable_fields.items():
            rows = [
                i for i, document in enumerate(documents)
                if document["fields"].get(field_name)
            ]

            if not rows:
                continue

            embeddings = self.embedder.encode_documents(
                [documents[i]["fields"][field_name] for i in rows]
            )

            vectors[rows] += weight * embeddings

        return vectors

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None
    ) -> list[dict]:
        """Return the records most similar to the query."""

        if self.index is None:
            raise RuntimeError("The index has not been built yet.")

        top_k = top_k or self.config.default_top_k

        if not 1 <= top_k <= self.config.max_top_k:
            raise ValueError(
                f"top_k must be between 1 and {self.config.max_top_k}."
            )

        if min_score is None:
            min_score = self.config.min_score

        hits = self.index.search(self.embedder.encode_query(query), top_k)

        return [
            {
                "id": hit["document_id"],
                "score": round(hit["score"], 4),
                "item": self.records[hit["document_id"]]
            }
            for hit in hits
            if hit["score"] >= min_score
        ]

    def load_or_build(
        self,
        dataset_path: str | Path | None = None,
        storage_dir: str | Path | None = None
    ) -> bool:
        """Load the saved index if it matches the dataset, otherwise rebuild it.

        Returns True when the index was rebuilt.
        """

        dataset_path = Path(dataset_path or self.config.dataset_path)
        storage_dir = Path(storage_dir or self.config.storage_dir)

        fingerprint = self._fingerprint(dataset_path)
        metadata = self._read_metadata(storage_dir)

        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("dimension") == self.embedder.dimension
        ):
            try:
                self._load(storage_dir)
                return False
            except (OSError, ValueError, RuntimeError) as error:
                logger.warning("Saved index is unusable, rebuilding: %s", error)

        self.build(load_records(dataset_path))
        self._save(storage_dir, fingerprint)

        return True

    def _fingerprint(self, dataset_path: Path) -> str:
        digest = hashlib.sha256(dataset_path.read_bytes())

        settings = {
            "id_field": self.config.id_field,
            "searchable_fields": self.config.searchable_fields,
            "model_name": self.config.model_name,
            "query_prefix": self.config.query_prefix,
            "document_prefix": self.config.document_prefix
        }

        digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))

        return digest.hexdigest()

    @staticmethod
    def _read_metadata(storage_dir: Path) -> dict:
        path = storage_dir / METADATA_FILE

        if not path.is_file():
            return {}

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, storage_dir: Path, fingerprint: str) -> None:
        self.index.save(storage_dir)

        (storage_dir / RECORDS_FILE).write_text(
            json.dumps(self.records, ensure_ascii=False),
            encoding="utf-8"
        )

        (storage_dir / METADATA_FILE).write_text(
            json.dumps({
                "fingerprint": fingerprint,
                "model_name": self.config.model_name,
                "dimension": self.embedder.dimension,
                "total_documents": self.total_documents
            }, indent=2),
            encoding="utf-8"
        )

    def _load(self, storage_dir: Path) -> None:
        index = VectorIndex.load(storage_dir)
        records = json.loads(
            (storage_dir / RECORDS_FILE).read_text(encoding="utf-8")
        )

        if set(index.document_ids) != set(records):
            raise ValueError("Saved records do not match the saved index.")

        self.index = index
        self.records = records

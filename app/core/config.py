
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _origins_from_env() -> list[str]:
    origins = os.environ.get(
        "ALLOWED_ORIGINS",
        ",".join(
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (3000, 4173, 5173, 5174)
        )
    )
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


@dataclass
class CollectionConfig:
    """Searchable fields for records that an application sends per request."""

    searchable_fields: dict[str, float]
    id_field: str = "id"


def _default_collections() -> dict[str, CollectionConfig]:
    return {
        "store-products": CollectionConfig(
            searchable_fields={
                "title": 3.0,
                "description": 2.0,
                "category": 1.0,
                "brand": 1.0
            }
        ),
        "delivery-meals": CollectionConfig(
            searchable_fields={
                "name": 3.0,
                "description": 2.0,
                "category": 1.0,
                "ingredients": 1.0
            }
        )
    }


@dataclass
class SearchConfig:
    id_field: str = "id"

    searchable_fields: dict[str, float] = field(
        default_factory=lambda: {
            "name": 3.0,
            "description": 2.0,
            "category": 1.0
        }
    )

    model_name: str = "intfloat/multilingual-e5-small"
    query_prefix: str = "query: "
    document_prefix: str = "passage: "

    dataset_path: Path = PROJECT_ROOT / "datasets" / "products.json"
    storage_dir: Path = PROJECT_ROOT / "storage"

    default_top_k: int = 5
    max_top_k: int = 50
    min_score: float = 0.0

    allowed_origins: list[str] = field(default_factory=_origins_from_env)

    collections: dict[str, CollectionConfig] = field(
        default_factory=_default_collections
    )
    max_items: int = 5000
    embedding_cache_size: int = 50000

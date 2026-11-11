
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import SearchConfig
from app.retrieval.item_search import ItemSearcher
from app.retrieval.semantic_search import SemanticSearchEngine

logging.basicConfig(level=logging.INFO)

config = SearchConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.embeddings.model import EmbeddingService

    embedder = EmbeddingService(
        model_name=config.model_name,
        query_prefix=config.query_prefix,
        document_prefix=config.document_prefix
    )

    engine = SemanticSearchEngine(config, embedder=embedder)
    engine.load_or_build()

    app.state.engine = engine
    app.state.item_searcher = ItemSearcher(
        config.collections,
        embedder,
        cache_size=config.embedding_cache_size
    )
    yield


app = FastAPI(title="Semantic Search Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=config.default_top_k, ge=1, le=config.max_top_k)


class ItemSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    items: list[dict[str, Any]] = Field(max_length=config.max_items)
    top_k: int | None = Field(default=None, ge=1, le=config.max_items)
    min_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_score_gap: float | None = Field(default=None, ge=0.0, le=2.0)


class SearchResult(BaseModel):
    id: str
    score: float
    item: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]


def run_search(request: Request, query: str, top_k: int) -> SearchResponse:
    engine: SemanticSearchEngine = request.app.state.engine

    try:
        results = engine.search(query, top_k=top_k)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return SearchResponse(query=query, total=len(results), results=results)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model": config.model_name,
        "indexed_documents": request.app.state.engine.total_documents,
        "collections": list(config.collections)
    }


@app.get("/collections")
def list_collections() -> dict:
    return {
        name: {
            "id_field": collection.id_field,
            "searchable_fields": collection.searchable_fields,
            "min_score": collection.min_score,
            "max_score_gap": collection.max_score_gap
        }
        for name, collection in config.collections.items()
    }


@app.get("/search", response_model=SearchResponse)
def search_get(
    request: Request,
    q: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=config.default_top_k, ge=1, le=config.max_top_k)
) -> SearchResponse:
    return run_search(request, q, top_k)


@app.post("/search", response_model=SearchResponse)
def search_post(request: Request, body: SearchRequest) -> SearchResponse:
    return run_search(request, body.query, body.top_k)


@app.post("/search/{collection}", response_model=SearchResponse)
def search_collection(
    collection: str,
    request: Request,
    body: ItemSearchRequest
) -> SearchResponse:
    """Rank the items an application sends, e.g. its current catalogue."""

    if collection not in config.collections:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown collection: {collection}"
        )

    searcher: ItemSearcher = request.app.state.item_searcher

    try:
        results = searcher.search(
            collection,
            body.query,
            body.items,
            top_k=body.top_k,
            min_score=body.min_score,
            max_score_gap=body.max_score_gap
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return SearchResponse(query=body.query, total=len(results), results=results)

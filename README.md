# Semantic Search Engine

Meaning-based product search (English + Arabic) built with
[sentence-transformers](https://www.sbert.net/) (`intfloat/multilingual-e5-small`),
[FAISS](https://github.com/facebookresearch/faiss) and [FastAPI](https://fastapi.tiangolo.com/).

## How it works

```
datasets/products.json
  -> app/data/loaders.py            load JSON records
  -> app/data/preprocessors.py      clean searchable fields (name, description, category)
  -> app/embeddings/model.py        embed each field (E5 "passage:" / "query:" prefixes)
  -> app/retrieval/semantic_search.py
       combine field vectors with the weights in config.py (name 3, description 2, category 1)
  -> app/indexing/vector_index.py   FAISS inner-product index (cosine similarity)
  -> storage/                       index saved here, rebuilt automatically when the dataset
                                    or search settings change
  -> app/main.py                    HTTP API
```

`USE_TF=0` is set inside `app/embeddings/model.py`, so TensorFlow/Keras is never loaded.

## Run it

Requirements: Python 3.10+ and git. No API keys or accounts are needed; the model is downloaded from Hugging Face on first start and then runs locally.

```bash
git clone https://github.com/reham0138/semantic-search-engine.git
cd semantic-search-engine

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

The first start downloads the model (~470 MB) and builds the index, so it can take a few minutes. Wait for `Application startup complete`, then check:

- http://127.0.0.1:8000/health returns `"status": "ok"`
- http://127.0.0.1:8000/docs has interactive API docs

Keep this terminal open while you run the store or delivery app.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## API

| Method | Path | Body / params | Description |
| ------ | ---- | ------------- | ----------- |
| GET | `/health` | - | status, model, number of indexed documents |
| GET | `/search` | `q` (required), `top_k` (1-50, default 5) | search |
| POST | `/search` | `{"query": "...", "top_k": 5}` | same search, JSON body |

Example:

```bash
curl "http://localhost:8000/search?q=something%20cheesy&top_k=3"
```

```json
{
  "query": "something cheesy",
  "total": 3,
  "results": [
    {"id": "P104", "score": 0.87, "item": {"id": "P104", "name": "Cheese Pizza", "category": "Pizza", "price": 6.0}}
  ]
}
```

`item` is the full original record, so any extra fields (price, image, slug...) come back as-is.

### Searching data sent by the app (`/search/{collection}`)

The store and delivery apps keep their catalogue in the browser (mock JSON plus localStorage edits), so they send the items they currently show and get them back ranked:

| Method | Path | Body | Description |
| ------ | ---- | ---- | ----------- |
| GET | `/collections` | - | collections with their weighted fields and cutoffs |
| POST | `/search/store-products` | `{"query": "...", "items": [...]}` | store products (`title` 3, `description` 2, `category` 1, `brand` 1) |
| POST | `/search/delivery-meals` | `{"query": "...", "items": [...]}` | meals (`name`/`localizedName` 3, `description` 2, `category`/`localizedCategory` 1, `ingredients` 1) |

Optional body fields: `top_k`, `min_score`, `max_score_gap`. By default a result must score at least `0.85` and be within `0.02` of the best match. multilingual-e5 scores unrelated text around 0.80-0.84, so this hides noise. Field embeddings are cached by text, so repeat searches only embed the query. Collections are defined in `app/core/config.py`.

Connected apps (`Desktop\apps`):

- `store-app-`: `frontend/src/shared/services/semanticSearch.service.ts` filters by branch stock, then calls `/search/store-products`.
- `delivey-app`: `frontend/src/shared/services/semanticSearch.service.ts` is used by `pages/Meals.tsx`. Name matches are shown first, and the page falls back to them if the API is down.

Both read `VITE_SEARCH_API_URL` (default `http://127.0.0.1:8000`).

## Using your own data

* Replace `datasets/products.json` with a JSON array of objects. Each object needs an `id`.
* Change which fields are searched (and how important each one is) in
  `SearchConfig.searchable_fields` in `app/core/config.py`.
* Restart the server: the index is rebuilt automatically when the file or settings change.

## Configuration

| Setting | Where | Default |
| ------- | ----- | ------- |
| Allowed browser origins (CORS) | `ALLOWED_ORIGINS` env var, comma-separated | `localhost` and `127.0.0.1` on ports 3000, 4173, 5173, 5174 |
| Model, prefixes, top_k limits, paths | `app/core/config.py` | see file |

---

# Using it from React / Next.js

The Python API runs as its own service (port 8000). Your frontend calls it over HTTP.
Start it first: `uvicorn app.main:app --reload --port 8000`.

## Shared TypeScript types

`types/search.ts`

```ts
export type Product = {
  id: string;
  name?: string | number;
  description?: string;
  category?: string;
  price?: number;
  [key: string]: unknown;
};

export type SearchResult = {
  id: string;
  score: number;
  item: Product;
};

export type SearchResponse = {
  query: string;
  total: number;
  results: SearchResult[];
};
```

## Next.js (App Router) - recommended setup

Call the Python API from a Next.js Route Handler. The browser only talks to your Next.js
app, so there is no CORS setup and the API URL stays private.

`.env.local`

```bash
SEARCH_API_URL=http://localhost:8000
```

`app/api/search/route.ts`

```ts
import { NextRequest, NextResponse } from "next/server";

const SEARCH_API_URL = process.env.SEARCH_API_URL ?? "http://localhost:8000";

export async function GET(request: NextRequest) {
  const q = request.nextUrl.searchParams.get("q")?.trim();
  const topK = request.nextUrl.searchParams.get("top_k") ?? "5";

  if (!q) {
    return NextResponse.json({ query: "", total: 0, results: [] });
  }

  try {
    const response = await fetch(
      `${SEARCH_API_URL}/search?${new URLSearchParams({ q, top_k: topK })}`,
      { cache: "no-store" }
    );
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { detail: "Search service is unavailable." },
      { status: 503 }
    );
  }
}
```

`hooks/useSemanticSearch.ts` (search while typing, with debounce and request cancelling)

```ts
"use client";

import { useEffect, useState } from "react";
import type { SearchResult } from "@/types/search";

export function useSemanticSearch(query: string, topK = 5, endpoint = "/api/search") {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setResults([]);
      setError(null);
      return;
    }

    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const params = new URLSearchParams({ q, top_k: String(topK) });
        const response = await fetch(`${endpoint}?${params}`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Search failed (${response.status})`);
        const data = await response.json();
        setResults(data.results);
      } catch (err) {
        if ((err as Error).name !== "AbortError") setError((err as Error).message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 300);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, topK, endpoint]);

  return { results, loading, error };
}
```

`components/ProductSearch.tsx`

```tsx
"use client";

import { useState } from "react";
import { useSemanticSearch } from "@/hooks/useSemanticSearch";

export default function ProductSearch() {
  const [query, setQuery] = useState("");
  const { results, loading, error } = useSemanticSearch(query);

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search products... / ابحث عن منتج"
      />
      {loading && <p>Searching...</p>}
      {error && <p>{error}</p>}
      <ul>
        {results.map(({ id, score, item }) => (
          <li key={id}>
            <strong>{String(item.name ?? id)}</strong>
            {item.description && <span> - {item.description}</span>}
            <small> ({(score * 100).toFixed(0)}% match)</small>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

Use it in any page, e.g. `app/page.tsx`:

```tsx
import ProductSearch from "@/components/ProductSearch";

export default function Home() {
  return <ProductSearch />;
}
```

### Server-side search (search results page)

For a `/search?q=...` page rendered on the server (good for SEO and shareable links):

`app/search/page.tsx`

```tsx
import type { SearchResponse } from "@/types/search";

export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { q = "" } = await searchParams;
  let data: SearchResponse = { query: q, total: 0, results: [] };

  if (q.trim()) {
    const response = await fetch(
      `${process.env.SEARCH_API_URL}/search?${new URLSearchParams({ q })}`,
      { cache: "no-store" }
    );
    if (response.ok) data = await response.json();
  }

  return (
    <ul>
      {data.results.map(({ id, item }) => (
        <li key={id}>{String(item.name ?? id)}</li>
      ))}
    </ul>
  );
}
```

(On Next.js 14 and older, `searchParams` is a plain object: use `searchParams.q` without `await`.)

## Plain React (Vite or Create React App)

React runs in the browser, so it calls the Python API directly. CORS already allows
`http://localhost:5173` (Vite) and `http://localhost:3000` (CRA).

`.env` (Vite)

```bash
VITE_SEARCH_API_URL=http://localhost:8000
```

Reuse the same `useSemanticSearch` hook (remove the `"use client"` line and change the
`@/types/search` import to a relative path), passing the Python endpoint:

```tsx
const API = import.meta.env.VITE_SEARCH_API_URL; // CRA: process.env.REACT_APP_SEARCH_API_URL
const { results, loading, error } = useSemanticSearch(query, 5, `${API}/search`);
```

If your React app runs on a different port or domain, start the API with that origin allowed:

```bash
# Windows cmd
set ALLOWED_ORIGINS=http://localhost:4200
uvicorn app.main:app --port 8000
```

## Deploying

1. Deploy the Python API as its own service (Render, Railway, Fly.io, a VPS, Docker...):
   `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. Next.js: set `SEARCH_API_URL` to the deployed API URL in your hosting provider.
3. Plain React: set `VITE_SEARCH_API_URL` to the API URL and set `ALLOWED_ORIGINS` on the
   API to your frontend's domain (e.g. `https://my-shop.vercel.app`).

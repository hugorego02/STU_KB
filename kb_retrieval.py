from __future__ import annotations

import json

import faiss
import numpy as np
from openai import OpenAI

from kb_config import EMB_OPENAI_MODEL, FAISS_PATH, MAX_CONTEXT_CHARS_PER_CHUNK, META_PATH


def load_kb():
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Index not found. Run first: python vectorize.py")

    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-12)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-12)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    response = client.embeddings.create(model=EMB_OPENAI_MODEL, input=texts)
    vectors = np.array([row.embedding for row in response.data], dtype="float32")
    return normalize_rows(vectors)


def get_query_embedding(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(model=EMB_OPENAI_MODEL, input=[text])
    vector = np.array(response.data[0].embedding, dtype="float32")
    return normalize(vector).reshape(1, -1)


def retrieve(query: str, index, meta, client: OpenAI, k: int, max_context_chars: int = MAX_CONTEXT_CHARS_PER_CHUNK):
    qvec = get_query_embedding(client, query)
    scores, ids = index.search(qvec, k)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        item = meta[int(idx)]
        full_text = item.get("text") or item.get("preview") or ""
        text = full_text[:max_context_chars]
        results.append({
            "score": float(score),
            "source_file": item.get("source_file", "unknown"),
            "chunk_id": item.get("chunk_id", "?"),
            "text": text,
            "preview": item.get("preview", ""),
        })
    return results


def rerank_filter_hits(hits: list[dict], max_per_doc: int = 2, max_total: int = 6) -> list[dict]:
    seen = set()
    per_doc = {}
    filtered = []

    for hit in hits:
        doc = hit.get("source_file", "unknown")
        per_doc.setdefault(doc, 0)
        if per_doc[doc] >= max_per_doc:
            continue

        text = (hit.get("text") or "").strip().lower()
        key = " ".join(text.split())[:300]
        if not key or key in seen:
            continue

        seen.add(key)
        filtered.append(hit)
        per_doc[doc] += 1

        if len(filtered) >= max_total:
            break

    return filtered

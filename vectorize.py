from __future__ import annotations

from pathlib import Path
import json
import os
import time

import numpy as np
import docx
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from kb_config import EMB_OPENAI_MODEL, FAISS_PATH, INDEX_DIR, META_PATH, RAW_DIR
from kb_retrieval import embed_texts

INDEX_DIR.mkdir(exist_ok=True)

# Chunking
MAX_CHARS = 1100
OVERLAP_PARAS = 1

# Batching (helps cost + speed)
BATCH_SIZE = 64
SLEEP_BETWEEN_BATCHES_SEC = 0.0  # set to 0.2 if you ever hit rate limits


# =========================
# DOCX ingestion
# =========================
def extract_paragraphs_docx(path: Path) -> list[str]:
    doc = docx.Document(str(path))
    paras: list[str] = []

    for p in doc.paragraphs:
        text = (p.text or "").strip()
        if not text:
            continue
        text = " ".join(text.split())
        paras.append(text)

    return paras


def chunk_by_paragraphs(paras: list[str], max_chars: int, overlap_paras: int) -> list[str]:
    """
    Build chunks by grouping paragraphs until max_chars is reached.
    Keeps paragraph boundaries intact.
    """
    if not paras:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    i = 0

    while i < len(paras):
        p = paras[i]
        add_len = len(p) + (2 if buffer else 0)  # "\n\n" between paragraphs

        if buffer_len + add_len <= max_chars:
            buffer.append(p)
            buffer_len += add_len
            i += 1
            continue

        if buffer:
            chunks.append("\n\n".join(buffer).strip())

            # overlap last N paragraphs
            if overlap_paras > 0:
                buffer = buffer[-overlap_paras:]
                buffer_len = sum(len(x) for x in buffer) + (2 * (len(buffer) - 1) if len(buffer) > 1 else 0)
            else:
                buffer = []
                buffer_len = 0
            continue

        # single huge paragraph fallback
        chunks.append(p)
        i += 1

    if buffer:
        chunks.append("\n\n".join(buffer).strip())

    return chunks


# =========================
# FAISS index
# =========================
def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(vectors)
    return index


# =========================
# Main
# =========================
def main():
    # Load .env locally. On Render, embeddings won't be generated; this is for local build.
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY not set. Put it in .env (local) or environment variables.")

    client = OpenAI()

    if not RAW_DIR.exists():
        raise SystemExit(f"❌ Pasta não encontrada: {RAW_DIR}")

    files = sorted(RAW_DIR.rglob("*.docx"))
    if not files:
        raise SystemExit(f"❌ Nenhum .docx encontrado em: {RAW_DIR}")

    all_chunks: list[str] = []
    meta: list[dict] = []
    global_chunk_id = 0

    for f in files:
        paras = extract_paragraphs_docx(f)
        chunks = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)

        for local_id, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            meta.append({
                "source_file": str(f),
                "chunk_id": local_id,
                "chunk_global_id": global_chunk_id,
                # keep preview small (helps memory). You can remove "text" if you want even lighter runtime.
                "preview": chunk[:220].replace("\n", " "),
                "text": chunk,  # optional: remove later to reduce kb_meta size
            })
            global_chunk_id += 1

    if not all_chunks:
        raise SystemExit("❌ Nenhum chunk gerado (documentos vazios?)")

    print(f"📄 DOCX: {len(files)} | 🔹 Chunks: {len(all_chunks)}")
    print(f"🧠 Embedding model: {EMB_OPENAI_MODEL}")

    # Embed in batches
    vectors_list: list[np.ndarray] = []
    t0 = time.time()

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        vecs = embed_texts(client, batch)
        vectors_list.append(vecs)

        done = min(i + BATCH_SIZE, len(all_chunks))
        print(f"  - embedded {done}/{len(all_chunks)} chunks")
        if SLEEP_BETWEEN_BATCHES_SEC > 0:
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    vectors = np.vstack(vectors_list).astype("float32")
    dt = time.time() - t0
    print(f"✅ Embeddings ready: shape={vectors.shape} in {dt:.1f}s")

    index = build_index(vectors)
    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Índice gerado com sucesso:")
    print(f" - {FAISS_PATH}")
    print(f" - {META_PATH}")
    print("ℹ️  Similaridade: cosine (IndexFlatIP + normalized embeddings)")


if __name__ == "__main__":
    main()

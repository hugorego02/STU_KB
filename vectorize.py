from pathlib import Path
import json
import numpy as np
import docx
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# Paths / Config
# =========================
BASE = Path(__file__).parent
RAW_DIR = BASE / "data_raw" / "docs_reais"
INDEX_DIR = BASE / "data_index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
MAX_CHARS = 1100
OVERLAP_PARAS = 1

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

        # normalize whitespace
        text = " ".join(text.split())
        paras.append(text)

    return paras


def chunk_by_paragraphs(
    paras: list[str],
    max_chars: int,
    overlap_paras: int
) -> list[str]:
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
                buffer_len = sum(len(x) for x in buffer) + (
                    2 * (len(buffer) - 1) if len(buffer) > 1 else 0
                )
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
    # IMPORTANT: cosine similarity via inner product + normalized embeddings
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# =========================
# Main
# =========================
def main():
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
                "text": chunk,
                "preview": chunk[:220].replace("\n", " "),
            })

            global_chunk_id += 1

    if not all_chunks:
        raise SystemExit("❌ Nenhum chunk gerado (documentos vazios?)")

    print(f"📄 DOCX: {len(files)} | 🔹 Chunks: {len(all_chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.array(embeddings, dtype="float32")

    index = build_index(vectors)

    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("✅ Índice gerado com sucesso:")
    print(f" - {FAISS_PATH}")
    print(f" - {META_PATH}")
    print("ℹ️  Similaridade: cosine (IndexFlatIP + normalized embeddings)")


if __name__ == "__main__":
    main()

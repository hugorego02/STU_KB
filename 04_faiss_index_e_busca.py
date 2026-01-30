from pathlib import Path
import json
import numpy as np
import docx
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
AMOSTRA = BASE / "data_raw" / "_amostra"
INDEX_DIR = BASE / "data_index"
INDEX_DIR.mkdir(exist_ok=True)

# To avoid mixing with the "real" index later, we'll save with the _amostra suffix
FAISS_PATH = INDEX_DIR / "kb_amostra.faiss"
META_PATH = INDEX_DIR / "kb_amostra_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHARS = 1100
OVERLAP_PARAS = 1


def extract_paragraphs_docx(path: Path) -> list[str]:
    d = docx.Document(str(path))
    paras = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            t = " ".join(t.split())
            paras.append(t)
    return paras


def chunk_by_paragraphs(paras: list[str], max_chars: int, overlap_paras: int) -> list[str]:
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    i = 0

    while i < len(paras):
        p = paras[i]
        add_len = len(p) + (2 if buf else 0)  # "\n\n" separator

        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
            i += 1
            continue

        if buf:
            chunks.append("\n\n".join(buf).strip())
            # overlap: repeat the last N paragraphs in the next chunk
            if overlap_paras > 0:
                buf = buf[-overlap_paras:]
                buf_len = sum(len(x) for x in buf) + (2 * (len(buf) - 1) if len(buf) > 1 else 0)
            else:
                buf, buf_len = [], 0
            continue

        # huge paragraph that doesn't fit by itself: keep it as a single chunk
        chunks.append(p)
        i += 1

    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product with normalized embeddings = cosine similarity
    index.add(vectors)
    return index


def main():
    if not AMOSTRA.exists():
        raise SystemExit(f"I could not find the folder: {AMOSTRA}")

    files = sorted(AMOSTRA.glob("*.docx"))
    if not files:
        raise SystemExit(f"No .docx found in: {AMOSTRA}")

    # 1) Extract + create chunks
    chunks: list[str] = []
    meta: list[dict] = []

    for f in files:
        paras = extract_paragraphs_docx(f)
        chs = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)

        for i, ch in enumerate(chs):
            chunks.append(ch)
            meta.append({
                "source_file": str(f),
                "chunk_id": i,
                "text": ch,  # store the full text to display later
                "preview": ch[:220].replace("\n", " ")
            })

    if not chunks:
        raise SystemExit("No text/chunk was generated. (Empty docs?)")

    print(f"Files: {len(files)} | Total chunks: {len(chunks)}")

    # 2) Embeddings
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.array(emb, dtype="float32")
    print(f"Embeddings: {vectors.shape} (chunks x dimension)")

    # 3) FAISS index
    index = build_index(vectors)
    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n✅ Index created and saved:")
    print(f" - {FAISS_PATH}")
    print(f" - {META_PATH}")

    # 4) Test search
    print("\n=== SEARCH TEST ===")
    query = input("Type a question (in PT or EN): ").strip()
    if not query:
        print("Empty question. Exiting.")
        return

    qvec = model.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec, dtype="float32")

    k = 5
    scores, ids = index.search(qvec, k)

    print(f"\nTop {k} results:")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        m = meta[int(idx)]
        print(f"\n#{rank}  score={float(score):.3f}")
        print(f"File: {m['source_file']}")
        print(f"Chunk: {m['chunk_id']}")
        print(f"Excerpt:\n{m['text'][:800]}")  # show up to 800 chars


if __name__ == "__main__":
    main()

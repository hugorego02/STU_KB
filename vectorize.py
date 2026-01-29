from pathlib import Path
import json
import numpy as np
import docx
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
RAW_DIR = BASE / "data_raw" / "docs_reais"
INDEX_DIR = BASE / "data_index"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

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
        add_len = len(p) + (2 if buf else 0)  # "\n\n" entre parágrafos

        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
            i += 1
            continue

        if buf:
            chunks.append("\n\n".join(buf).strip())

            # overlap: repete os últimos N parágrafos no próximo chunk
            if overlap_paras > 0:
                buf = buf[-overlap_paras:]
                buf_len = sum(len(x) for x in buf) + (2 * (len(buf) - 1) if len(buf) > 1 else 0)
            else:
                buf, buf_len = [], 0
            continue

        # parágrafo gigante que não cabe no max_chars: vira chunk sozinho
        chunks.append(p)
        i += 1

    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks


def build_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity com embeddings normalizados
    index.add(vectors)
    return index


def main():
    if not RAW_DIR.exists():
        raise SystemExit(f"Não encontrei a pasta: {RAW_DIR}")

    files = sorted(RAW_DIR.rglob("*.docx"))
    if not files:
        raise SystemExit(f"Nenhum .docx encontrado em: {RAW_DIR}")

    all_chunks: list[str] = []
    meta: list[dict] = []

    for f in files:
        paras = extract_paragraphs_docx(f)
        chunks = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)

        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "source_file": str(f),
                "chunk_id": i,
                "text": ch,  # por enquanto guardamos texto completo (depois podemos otimizar)
                "preview": ch[:220].replace("\n", " ")
            })

    if not all_chunks:
        raise SystemExit("Nenhum chunk gerado. (Docs vazios?)")

    print(f"Arquivos DOCX: {len(files)} | Chunks: {len(all_chunks)}")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(all_chunks, normalize_embeddings=True, show_progress_bar=True)
    vectors = np.array(emb, dtype="float32")

    index = build_index(vectors)

    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Índice REAL gerado:")
    print(f" - {FAISS_PATH}")
    print(f" - {META_PATH}")


if __name__ == "__main__":
    main()

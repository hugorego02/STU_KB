from pathlib import Path
import numpy as np
import docx
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
AMOSTRA = BASE / "data_raw" / "_amostra"

MAX_CHARS = 1100
OVERLAP_PARAS = 1

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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

    chunks = []
    buf = []
    buf_len = 0
    i = 0

    while i < len(paras):
        p = paras[i]
        add_len = len(p) + (2 if buf else 0)

        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
            i += 1
            continue

        if buf:
            chunks.append("\n\n".join(buf))

            if overlap_paras > 0:
                buf = buf[-overlap_paras:]
                buf_len = sum(len(x) for x in buf)
            else:
                buf = []
                buf_len = 0
            continue

        chunks.append(p)
        i += 1

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def main():
    arquivos = sorted(AMOSTRA.glob("*.docx"))
    if not arquivos:
        raise SystemExit("No .docx found.")

    # pick the largest file for testing
    arq = max(arquivos, key=lambda p: p.stat().st_size)
    print(f"Test file: {arq.name}")

    paras = extract_paragraphs_docx(arq)
    chunks = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)

    print(f"Generated chunks: {len(chunks)}")

    # load embeddings model
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(f"Generated embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings[0].shape[0]}")

    # compare similarity between some chunks
    if len(embeddings) >= 2:
        s01 = cosine_sim(embeddings[0], embeddings[1])
        print(f"Similarity chunk 1 vs 2: {s01:.3f}")

    if len(embeddings) >= 3:
        s02 = cosine_sim(embeddings[0], embeddings[2])
        print(f"Similarity chunk 1 vs 3: {s02:.3f}")


if __name__ == "__main__":
    main()

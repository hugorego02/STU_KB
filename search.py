from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
INDEX_DIR = BASE / "data_index"

FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Índice não encontrado. Rode primeiro: python vectorize.py")

    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    model = SentenceTransformer(MODEL_NAME)

    while True:
        q = input("\nPergunta (ENTER para sair): ").strip()
        if not q:
            break

        qvec = model.encode([q], normalize_embeddings=True)
        qvec = np.array(qvec, dtype="float32")

        k = 5
        scores, ids = index.search(qvec, k)

        print(f"\nTop {k} resultados:")
        for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if idx < 0:
                continue
            m = meta[int(idx)]
            print(f"\n#{rank}  score={float(score):.3f}")
            print(f"Arquivo: {m['source_file']}")
            print(f"Chunk:  {m['chunk_id']}")
            print(f"Trecho:\n{m['text'][:700]}")


if __name__ == "__main__":
    main()

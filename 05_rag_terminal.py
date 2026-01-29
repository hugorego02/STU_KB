from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --------- Config ----------
BASE = Path(__file__).parent
INDEX_DIR = BASE / "data_index"
FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"  # bom custo/latência para MVP (pode trocar depois)

TOP_K = 5
MAX_CONTEXT_CHARS_PER_CHUNK = 900  # para não mandar chunks gigantes pro LLM

SYSTEM_RULES = """\
You are an assistant. Answer using ONLY the SOURCES provided.
- If the answer is not contained in the sources, say you could not find it in the knowledge base.
- Be concise and actionable.
- At the end, list sources used in this format: (Filename - Chunk).
"""

# --------- Helpers ----------
def load_kb():
    if not FAISS_PATH.exists() or not META_PATH.exists():
        raise SystemExit("Índice não encontrado. Rode primeiro: python vectorize.py")

    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta


def retrieve(query: str, index: faiss.Index, meta: list[dict], emb_model: SentenceTransformer, k: int = TOP_K):
    qvec = emb_model.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec, dtype="float32")

    scores, ids = index.search(qvec, k)
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        text = m.get("text", "")
        results.append({
            "score": float(score),
            "source_file": m["source_file"],
            "chunk_id": m["chunk_id"],
            "text": text[:MAX_CONTEXT_CHARS_PER_CHUNK],
        })
    return results


def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[SOURCE]\nfile: {r['source_file']}\nchunk: {r['chunk_id']}\nscore: {r['score']:.3f}\ntext:\n{r['text']}\n"
        )
    return "\n".join(blocks)


def main():
    load_dotenv()  # carrega OPENAI_API_KEY do .env
    client = OpenAI()

    index, meta = load_kb()
    emb_model = SentenceTransformer(EMB_MODEL)

    print("RAG pronto. (ENTER vazio para sair)\n")

    while True:
        q = input("Pergunta: ").strip()
        if not q:
            break

        results = retrieve(q, index, meta, emb_model, k=TOP_K)
        context = build_context(results)

        user_prompt = f"""\
Pergunta do usuário:
{q}

FONTES (use somente isso):
{context}
"""

        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": user_prompt},
            ],
        )

        # A forma mais simples de pegar texto:
        answer = resp.output_text
        print("\n--- RESPOSTA ---")
        print(answer)

        print("\n--- FONTES RECUPERADAS (top-k) ---")
        for i, r in enumerate(results, start=1):
            print(f"{i}) score={r['score']:.3f} | {Path(r['source_file']).name} - chunk {r['chunk_id']}")
        print()


if __name__ == "__main__":
    main()

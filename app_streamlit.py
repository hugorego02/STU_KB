from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE = Path(__file__).parent
INDEX_DIR = BASE / "data_index"
FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5
MAX_CONTEXT_CHARS_PER_CHUNK = 900

SYSTEM_RULES = """\
Você é um assistente interno que responde usando APENAS as FONTES fornecidas.
- Se a resposta não estiver nas fontes, diga claramente que não encontrou na base.
- Seja objetivo e prático.
- No final, liste as fontes usadas no formato: (Arquivo - Chunk).
"""

@st.cache_resource
def load_models_and_kb():
    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    emb_model = SentenceTransformer(EMB_MODEL)
    return index, meta, emb_model

def retrieve(query: str, index, meta, emb_model, k=TOP_K):
    qvec = emb_model.encode([query], normalize_embeddings=True)
    qvec = np.array(qvec, dtype="float32")
    scores, ids = index.search(qvec, k)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        results.append({
            "score": float(score),
            "source_file": m["source_file"],
            "chunk_id": m["chunk_id"],
            "text": (m.get("text", "")[:MAX_CONTEXT_CHARS_PER_CHUNK]),
            "preview": m.get("preview", "")
        })
    return results

def build_context(results):
    blocks = []
    for r in results:
        blocks.append(
            f"[SOURCE]\nfile: {r['source_file']}\nchunk: {r['chunk_id']}\nscore: {r['score']:.3f}\ntext:\n{r['text']}\n"
        )
    return "\n".join(blocks)

def main():
    load_dotenv()
    client = OpenAI()

    st.set_page_config(page_title="STU KB RAG", layout="wide")
    st.title("STU Knowledge Base Chat (RAG)")

    if not FAISS_PATH.exists() or not META_PATH.exists():
        st.error("Índice não encontrado. Rode primeiro: python vectorize.py")
        st.stop()

    index, meta, emb_model = load_models_and_kb()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Faça uma pergunta…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

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
        answer = resp.output_text

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

            with st.expander("Fontes (top-k)"):
                for r in results:
                    st.write(f"score={r['score']:.3f} | {Path(r['source_file']).name} - chunk {r['chunk_id']}")
                    st.caption(r["preview"])

if __name__ == "__main__":
    main()

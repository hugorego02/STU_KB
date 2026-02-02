from __future__ import annotations

from pathlib import Path
import json
import re
import time
import os

import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Paths / Config
# =========================
BASE = Path(__file__).parent
INDEX_DIR = BASE / "data_index"
FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

# LLM used for answering + summarizing
LLM_MODEL = "gpt-4o-mini"

# Embeddings used for retrieval (MUST match what you used in vectorize.py)
EMB_OPENAI_MODEL = "text-embedding-3-small"

TOP_K = 6
MAX_CONTEXT_CHARS_PER_CHUNK = 900

# Confidence gating (IndexFlatIP + normalized vectors => cosine similarity)
MIN_TOP1 = 0.35
MIN_TOP1_STRONG = 0.40
MIN_GAP_12 = 0.03

SUMMARY_EVERY_TURNS = 6

SOURCE_TRIGGER_PATTERNS = [
    r"\bsource(s)?\b",
    r"\breference(s)?\b",
    r"\bcitation(s)?\b",
    r"\bwhich document\b",
    r"\bwhere is this from\b",
    r"\bshow sources\b",
    r"\bqual( é| e)?(m)? a(s)? fonte(s)?\b",
    r"\bde qual arquivo\b",
    r"\bonde isso está\b",
    r"\bmostra(r)? (as )?fontes\b",
    r"\bmostrar (as )?fontes\b",
    r"\bfontes\??$",
    r"\bfonte\??$",
]

RESET_PATTERNS = [
    r"^/reset$",
    r"^reset$",
    r"^/clear$",
    r"^clear$",
    r"^/new$",
    r"^new$",
]

# =========================
# Prompting
# =========================
SYSTEM_PROMPT_KB = """\
You are an internal STU support assistant.

Style:
- Reply in a concise, natural chat tone (2–6 short lines).
- Use short paragraphs. Use bullets only for steps.
- Do NOT sound like documentation.

Answer format:
- First line: the direct answer in one sentence.
- If applicable: 2–5 bullet steps.

Evidence rules:
- Use ONLY the provided excerpts as evidence.
- User context is NOT evidence. It is only to interpret the question.
- Do NOT invent steps, policies, tools, owners, approvals, or responsibilities.
- If the excerpts do not contain the answer, say: "I couldn’t find this in the current documents."

Sources:
- Do NOT display sources unless the user explicitly asks for them.
"""

SYSTEM_PROMPT_CHAT = """\
You are a friendly internal assistant.

Rules:
- Chat naturally (short, helpful, human).
- You can do small talk, greetings, and general conversation.
- Keep it concise.
- Do not mention documents or knowledge bases unless the user asks.
"""

SUMMARIZER_SYSTEM = "You are a summarizer. Be brief and factual."

# =========================
# Helpers
# =========================
def user_asked_for_sources(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return any(re.search(p, t) for p in SOURCE_TRIGGER_PATTERNS)

def is_sources_only_message(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if not user_asked_for_sources(t):
        return False
    sources_only_phrases = {
        "sources", "source", "show sources",
        "fontes", "fonte",
        "mostra as fontes", "mostrar fontes", "mostrar as fontes",
        "qual a fonte", "quais as fontes",
        "de qual arquivo", "onde isso está",
    }
    return t in sources_only_phrases or len(t) <= 30

def is_reset_cmd(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return any(re.search(p, t) for p in RESET_PATTERNS)

def format_answer_natural(answer: str) -> str:
    if not answer:
        return answer
    a = answer.strip()
    a = re.sub(r"\n{3,}", "\n\n", a)
    return a

def is_inner_product_index(index) -> bool:
    mt = getattr(index, "metric_type", None)
    return mt == faiss.METRIC_INNER_PRODUCT

def build_sources_block(hits: list[dict], max_items: int = 5) -> str:
    if not hits:
        return ""
    lines = []
    for h in hits[:max_items]:
        file_ = Path(h.get("source_file", "Unknown file")).name
        chunk = h.get("chunk_id", "?")
        preview = (h.get("text", "") or "").strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "…"
        lines.append(f"- **{file_}** (chunk {chunk}) — {preview}")
    return "\n".join(lines)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "turns_since_summary" not in st.session_state:
        st.session_state.turns_since_summary = 0
    if "last_hits" not in st.session_state:
        st.session_state.last_hits = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

# =========================
# Fast startup: KB only
# =========================
@st.cache_resource
def load_kb_only():
    index = faiss.read_index(str(FAISS_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return index, meta

def normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / (n + 1e-12)

def get_query_embedding(client: OpenAI, text: str) -> np.ndarray:
    # Returns shape (1, d) float32, normalized for cosine similarity.
    r = client.embeddings.create(
        model=EMB_OPENAI_MODEL,
        input=[text],
    )
    vec = np.array(r.data[0].embedding, dtype="float32")
    vec = normalize(vec).reshape(1, -1)
    return vec

def retrieve(query: str, index, meta, client: OpenAI, k=TOP_K):
    qvec = get_query_embedding(client, query)
    scores, ids = index.search(qvec, k)

    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        # Prefer preview to reduce memory pressure (if your meta has both)
        text = (m.get("preview") or m.get("text") or "")[:MAX_CONTEXT_CHARS_PER_CHUNK]
        results.append({
            "score": float(score),
            "source_file": m.get("source_file", "unknown"),
            "chunk_id": m.get("chunk_id", "?"),
            "text": text,
            "preview": m.get("preview", "")
        })
    return results

def rerank_filter_hits(hits: list[dict], max_per_doc: int = 2, max_total: int = TOP_K) -> list[dict]:
    seen = set()
    per_doc = {}
    out = []

    for h in hits:
        doc = h.get("source_file", "unknown")
        per_doc.setdefault(doc, 0)
        if per_doc[doc] >= max_per_doc:
            continue

        text = (h.get("text") or "").strip().lower()
        key = re.sub(r"\s+", " ", text)[:300]
        if not key or key in seen:
            continue
        seen.add(key)

        out.append(h)
        per_doc[doc] += 1

        if len(out) >= max_total:
            break

    return out

def build_excerpts_block(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            "-----\n"
            f"[{i}] File: {Path(r['source_file']).name} | Chunk: {r['chunk_id']}\n"
            f"{r['text']}\n"
        )
    return "\n".join(blocks)

def should_answer(hits: list[dict]) -> tuple[bool, str]:
    if not hits:
        return False, "no_hits"

    top1 = hits[0]["score"]
    top2 = hits[1]["score"] if len(hits) > 1 else 0.0
    gap12 = top1 - top2

    if top1 < MIN_TOP1:
        return False, "top1_low"

    if top1 < MIN_TOP1_STRONG and gap12 < MIN_GAP_12:
        return False, "ambiguous_low_gap"

    return True, "ok"

def maybe_update_summary(client: OpenAI):
    if st.session_state.turns_since_summary < SUMMARY_EVERY_TURNS:
        return

    last_msgs = st.session_state.messages[-12:]
    convo_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in last_msgs])

    prompt = f"""\
Update the running conversation summary.
Keep it short (max 6 bullet points).
Include user intent, constraints, and any confirmed details.
Avoid quotes.

Current summary:
{st.session_state.summary or "(none)"}

Recent conversation:
{convo_text}
"""

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )

    st.session_state.summary = (resp.output_text or "").strip()
    st.session_state.turns_since_summary = 0

def build_llm_input_kb(question: str, excerpts: str) -> str:
    summary = st.session_state.summary or "(none)"
    return f"""\
User context (for interpreting the question; NOT evidence):
{summary}

User question:
{question}

Excerpts (evidence — use only this):
{excerpts}
"""

def build_llm_input_chat(user_text: str) -> str:
    summary = st.session_state.summary or "(none)"
    recent = st.session_state.messages[-10:]
    recent_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in recent]) or "(none)"
    return f"""\
Conversation summary:
{summary}

Recent chat:
{recent_text}

User message:
{user_text}
"""

def should_use_kb(text: str) -> bool:
    """
    Fast router for best performance (no extra LLM call).
    If it looks like STU/process/system/procedure -> use KB.
    Otherwise -> normal chat.
    """
    t = (text or "").lower()

    kb_keywords = [
        "stu", "bobcat", "portal",
        "canvas", "servicenow", "snow",
        "password", "reset", "mfa",
        "wifi", "mac address", "vpn",
        "incident", "request", "ticket",
        "azure", "ad", "active directory",
        "ellucian", "insights",
        "ipad", "pearson", "lockdown",
        "meeting room", "service desk"
    ]
    kb_triggers = ["how do i", "how to", "where", "setup", "install", "access", "login", "create an incident"]

    return any(k in t for k in kb_keywords) or any(x in t for x in kb_triggers)

# =========================
# App
# =========================
def main():
    load_dotenv()

    st.set_page_config(page_title="STU KB Support Chat", layout="wide")
    st.title("STU Knowledge Base Support Chat")

    # Fail fast if index missing
    if not FAISS_PATH.exists() or not META_PATH.exists():
        st.error("Index not found. Run first: python vectorize.py")
        st.stop()

    # Fail fast if OPENAI_API_KEY missing
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it in Render → Environment.")
        st.stop()

    client = OpenAI()

    # Load: FAISS + meta only (fast + low memory)
    index, meta = load_kb_only()
    init_session_state()

    with st.sidebar:
        st.markdown("### Controls")
        debug_show_sources = st.toggle("Debug: always show sources", value=False)
        debug_show_retrieval = st.toggle("Debug: show retrieved previews", value=False)

        st.markdown("---")
        metric = "INNER_PRODUCT (cosine similarity; vectors normalized)" if is_inner_product_index(index) else "UNKNOWN / not INNER_PRODUCT"
        st.caption(f"FAISS metric: {metric}")
        st.caption(f"Gating: top1>={MIN_TOP1:.2f} (or >= {MIN_TOP1_STRONG:.2f}) and gap12>={MIN_GAP_12:.2f}")

        if st.button("Reset conversation"):
            st.session_state.messages = []
            st.session_state.summary = ""
            st.session_state.turns_since_summary = 0
            st.session_state.last_hits = []
            st.session_state.last_question = ""
            st.rerun()

        st.markdown("---")
        st.caption("Tip: Ask “show sources” / “fontes” to reveal references.")

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask anything (general chat or STU procedures)…")
    if not q:
        return

    # Reset command
    if is_reset_cmd(q):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.turns_since_summary = 0
        st.session_state.last_hits = []
        st.session_state.last_question = ""
        st.rerun()

    # Sources-only message
    if is_sources_only_message(q):
        with st.chat_message("user"):
            st.markdown(q)
        st.session_state.messages.append({"role": "user", "content": q})

        hits = st.session_state.last_hits
        if hits:
            msg = "Here are the sources I used last time:\n\n" + build_sources_block(hits)
        else:
            msg = "I don’t have sources to show yet. Ask a STU/process question first."
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        return

    # Save user message + update memory counters
    st.session_state.messages.append({"role": "user", "content": q})
    st.session_state.turns_since_summary += 1
    st.session_state.last_question = q

    with st.chat_message("user"):
        st.markdown(q)

    # =========================
    # Mode 1: Normal chat (NO KB)
    # =========================
    if not should_use_kb(q):
        chat_input = build_llm_input_chat(q)
        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT_CHAT},
                {"role": "user", "content": chat_input},
            ],
        )
        answer = format_answer_natural(resp.output_text or "")
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        maybe_update_summary(client)
        return

    # =========================
    # Mode 2: STU KB (RAG)
    # =========================
    t0 = time.time()
    raw_hits = retrieve(q, index, meta, client, k=max(TOP_K * 3, TOP_K))
    hits = rerank_filter_hits(raw_hits, max_per_doc=2, max_total=TOP_K)
    t_retrieve = time.time() - t0
    st.session_state.last_hits = hits

    if not hits:
        answer = "I couldn’t find this in the current documents."
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        maybe_update_summary(client)
        return

    ok, reason = should_answer(hits)
    if not ok:
        answer = (
            "I couldn’t find this clearly in the current documents.\n\n"
            "If you share the exact process name, system/app name, or the form name, I can try again."
        )
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

            if debug_show_retrieval:
                with st.expander("Debug: why I refused"):
                    st.write(f"reason={reason}")
                    st.write(f"top1={hits[0]['score']:.3f}")
                    if len(hits) > 1:
                        st.write(f"top2={hits[1]['score']:.3f} | gap={hits[0]['score']-hits[1]['score']:.3f}")

        maybe_update_summary(client)
        return

    excerpts = build_excerpts_block(hits)
    kb_input = build_llm_input_kb(q, excerpts)

    t1 = time.time()
    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_KB},
            {"role": "user", "content": kb_input},
        ],
    )
    t_llm = time.time() - t1

    answer = format_answer_natural(resp.output_text or "")

    if user_asked_for_sources(q) or debug_show_sources:
        answer += "\n\n**Sources:**\n" + build_sources_block(hits)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        if debug_show_retrieval:
            with st.expander("Debug: retrieved chunks"):
                st.caption(f"Retrieval time: {t_retrieve:.3f}s | LLM time: {t_llm:.3f}s")
                for r in hits:
                    st.write(f"score={r['score']:.3f} | {Path(r['source_file']).name} — chunk {r['chunk_id']}")
                    if r.get("preview"):
                        st.caption(r["preview"])

    maybe_update_summary(client)

if __name__ == "__main__":
    main()

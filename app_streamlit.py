from __future__ import annotations

from pathlib import Path
import os
import re
import time

import faiss
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from kb_config import FAISS_PATH, LLM_MODEL, META_PATH, TOP_K, WEB_SEARCH_MODEL
from kb_retrieval import load_kb, rerank_filter_hits, retrieve

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

KB_META_PATTERNS = [
    r"\bknowledge base\b",
    r"\bkb\b",
    r"\bdatabase\b",
    r"\bdocuments?\b",
    r"\bdoc(s)?\b",
    r"\bsources?\b",
    r"\bfiles?\b",
]

KB_LIST_PATTERNS = [
    r"\blist\b.*\b(knowledge base|kb|documents|docs|files|sources)\b",
    r"\bwhat kb\b",
    r"\bwhich kb\b",
    r"\bwhat documents\b",
    r"\bwhich documents\b",
    r"\bshow\b.*\b(knowledge base|kb|documents|docs|files)\b",
    r"\bhave access\b",
]

RESET_PATTERNS = [
    r"^/reset$",
    r"^reset$",
    r"^/clear$",
    r"^clear$",
    r"^/new$",
    r"^new$",
]

FULL_KB_PATTERNS = [
    r"\bshow\b.*\bfull\b",
    r"\bfull text\b",
    r"\bwhole\b.*\bkb\b",
    r"\bentire\b.*\bkb\b",
    r"\ball write down\b",
    r"\bsend to me\b",
    r"\bpaste\b.*\bkb\b",
]

SYSTEM_PROMPT_KB = """\
You are an internal STU support assistant.

Use this order of priority:
1. STU KB excerpts provided in the prompt
2. Web search results, if needed
3. General reasoning only to connect the evidence, never to invent STU-specific facts

Rules:
- Always check the KB excerpts first.
- If the KB excerpts answer the question, prefer them.
- If the KB excerpts are incomplete, ambiguous, or unrelated, use web search to improve the answer.
- For STU-specific procedures, policies, phone numbers, portals, and internal processes, prefer KB information over the web.
- If neither KB nor web results support the answer, say that clearly.
- If the question is simple small talk, answer naturally and briefly.

Style:
- Reply in a concise, natural chat tone.
- Use short paragraphs. Use bullets only for steps or lists.
- Do not sound like documentation.
"""

SUMMARIZER_SYSTEM = "You are a summarizer. Be brief and factual."


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


def is_kb_meta_question(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return any(re.search(p, t) for p in KB_META_PATTERNS)


def is_kb_list_question(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    return any(re.search(p, t) for p in KB_LIST_PATTERNS)


def extract_kb_code(user_text: str) -> str | None:
    match = re.search(r"\b(KB\d{7})\b", user_text or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def is_full_kb_request(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    has_full_pattern = any(re.search(p, t) for p in FULL_KB_PATTERNS)
    has_kb_code = extract_kb_code(t) is not None
    return has_kb_code and has_full_pattern


def format_answer_natural(answer: str) -> str:
    if not answer:
        return answer
    return re.sub(r"\n{3,}", "\n\n", answer.strip())


def is_inner_product_index(index) -> bool:
    return getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT


def build_sources_block(hits: list[dict], max_items: int = 5) -> str:
    if not hits:
        return ""
    lines = []
    for hit in hits[:max_items]:
        file_name = os.path.basename(hit.get("source_file", "Unknown file"))
        chunk_id = hit.get("chunk_id", "?")
        preview = (hit.get("text", "") or "").strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        lines.append(f"- **{file_name}** (chunk {chunk_id}) - {preview}")
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


def list_kb_documents(meta: list[dict], max_items: int = 20) -> list[str]:
    names = sorted({Path(item.get("source_file", "unknown")).name for item in meta})
    return names[:max_items]


def build_kb_meta_answer(question: str, meta: list[dict]) -> str:
    docs = list_kb_documents(meta)
    total_docs = len({item.get("source_file", "unknown") for item in meta})
    total_chunks = len(meta)

    if is_kb_list_question(question):
        doc_lines = "\n".join(f"- {name}" for name in docs)
        more = ""
        if total_docs > len(docs):
            more = f"\n\nI have access to {total_docs} KB documents total, so this is the first {len(docs)}."
        return (
            "I’m using the indexed STU knowledge base for answers.\n\n"
            f"Here are some KB documents I can access:\n{doc_lines}{more}"
        )

    return (
        f"Yes. I answer questions using an indexed STU knowledge base with {total_docs} documents "
        f"and {total_chunks} searchable chunks.\n\n"
        "If you want, I can also list the KB files I currently have access to."
    )


def find_kb_documents(meta: list[dict], kb_code: str) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for item in meta:
        source_file = item.get("source_file", "unknown")
        file_name = Path(source_file).name.upper()
        if kb_code in file_name:
            grouped.setdefault(source_file, []).append(item)

    ordered = []
    for source_file, items in grouped.items():
        ordered.append((source_file, sorted(items, key=lambda x: int(x.get("chunk_id", 0)))))
    ordered.sort(key=lambda pair: Path(pair[0]).name)
    return ordered


def build_full_kb_answer(question: str, meta: list[dict], max_chars: int = 12000) -> str:
    kb_code = extract_kb_code(question)
    if not kb_code:
        return "Please include the KB code, for example `KB0010021`, and I’ll show the full indexed text."

    matches = find_kb_documents(meta, kb_code)
    if not matches:
        return f"I couldn’t find a KB with code {kb_code} in the indexed documents."

    source_file, chunks = matches[0]
    full_text = "\n\n".join((chunk.get("text") or "").strip() for chunk in chunks if (chunk.get("text") or "").strip())
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    if not full_text:
        return f"I found {Path(source_file).name}, but there is no indexed text available for it."

    truncated = ""
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars].rstrip()
        truncated = "\n\n[Truncated to fit the chat window. Ask me to continue if you want the rest.]"

    return f"Here is the indexed text for **{Path(source_file).name}**:\n\n{full_text}{truncated}"


@st.cache_resource
def load_kb_only():
    return load_kb()


def build_excerpts_block(results: list[dict]) -> str:
    if not results:
        return "(No relevant KB excerpts were retrieved.)"

    blocks = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            "-----\n"
            f"[{i}] File: {Path(item['source_file']).name} | Chunk: {item['chunk_id']}\n"
            f"{item['text']}\n"
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

KB excerpts:
{excerpts}
"""


def answer_with_kb_and_web(client: OpenAI, question: str, excerpts: str):
    kb_input = build_llm_input_kb(question, excerpts)
    return client.responses.create(
        model=WEB_SEARCH_MODEL,
        tools=[{"type": "web_search"}],
        tool_choice="auto",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_KB},
            {"role": "user", "content": kb_input},
        ],
    )


def answer_kb_only(client: OpenAI, question: str, excerpts: str):
    kb_input = build_llm_input_kb(question, excerpts)
    return client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT_KB},
            {"role": "user", "content": kb_input},
        ],
    )


def main():
    load_dotenv()

    st.set_page_config(page_title="STU KB Support Chat", layout="wide")
    st.title("STU Knowledge Base Support Chat")

    if not FAISS_PATH.exists() or not META_PATH.exists():
        st.error("Index not found. Run first: python vectorize.py")
        st.stop()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Add it in your environment.")
        st.stop()

    client = OpenAI()
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
        st.caption("The app always checks the KB first, then uses web search if needed.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything...")
    if not question:
        return

    if is_reset_cmd(question):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.turns_since_summary = 0
        st.session_state.last_hits = []
        st.session_state.last_question = ""
        st.rerun()

    if is_sources_only_message(question):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        hits = st.session_state.last_hits
        if hits:
            answer = "Here are the KB sources I used last time:\n\n" + build_sources_block(hits)
        else:
            answer = "I don’t have KB sources to show yet. Ask a question first."

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        return

    if is_full_kb_request(question):
        answer = build_full_kb_answer(question, meta)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.turns_since_summary += 1
        maybe_update_summary(client)
        return

    if is_kb_meta_question(question):
        answer = build_kb_meta_answer(question, meta)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.turns_since_summary += 1
        maybe_update_summary(client)
        return

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.turns_since_summary += 1
    st.session_state.last_question = question

    with st.chat_message("user"):
        st.markdown(question)

    t0 = time.time()
    raw_hits = retrieve(question, index, meta, client, k=max(TOP_K * 3, TOP_K))
    hits = rerank_filter_hits(raw_hits, max_per_doc=2, max_total=TOP_K)
    t_retrieve = time.time() - t0
    st.session_state.last_hits = hits

    excerpts = build_excerpts_block(hits)
    kb_ok, kb_reason = should_answer(hits)

    t1 = time.time()
    web_error = None
    try:
        resp = answer_with_kb_and_web(client, question, excerpts)
    except Exception as exc:
        web_error = str(exc)
        resp = answer_kb_only(client, question, excerpts)
    t_llm = time.time() - t1

    answer = format_answer_natural(resp.output_text or "")

    if user_asked_for_sources(question) or debug_show_sources:
        sources_block = build_sources_block(hits)
        if sources_block:
            answer += "\n\n**KB Sources:**\n" + sources_block
        else:
            answer += "\n\n**KB Sources:**\n- No relevant KB excerpts were retrieved for this question."

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

        if debug_show_retrieval:
            with st.expander("Debug: retrieval details"):
                st.caption(f"Retrieval time: {t_retrieve:.3f}s | LLM time: {t_llm:.3f}s")
                st.write(f"kb_confidence={kb_reason}")
                st.write(f"kb_has_clear_match={kb_ok}")
                if web_error:
                    st.write(f"web_fallback_error={web_error}")
                for hit in hits:
                    st.write(f"score={hit['score']:.3f} | {Path(hit['source_file']).name} - chunk {hit['chunk_id']}")
                    if hit.get("preview"):
                        st.caption(hit["preview"])

    maybe_update_summary(client)


if __name__ == "__main__":
    main()

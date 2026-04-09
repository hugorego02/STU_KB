from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).parent
RAW_DIR = BASE / "data_raw" / "docs_reais"
INDEX_DIR = BASE / "data_index"

FAISS_PATH = INDEX_DIR / "kb.faiss"
META_PATH = INDEX_DIR / "kb_meta.json"

EMB_OPENAI_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
WEB_SEARCH_MODEL = "gpt-5"

TOP_K = 6
MAX_CONTEXT_CHARS_PER_CHUNK = 900

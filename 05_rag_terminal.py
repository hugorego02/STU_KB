from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from kb_config import LLM_MODEL, TOP_K
from kb_retrieval import load_kb, retrieve

SYSTEM_RULES = """\
You are an assistant. Answer using ONLY the SOURCES provided.
- If the answer is not contained in the sources, say you could not find it in the knowledge base.
- Be concise and actionable.
- At the end, list sources used in this format: (Filename - Chunk).
"""

def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[SOURCE]\nfile: {r['source_file']}\nchunk: {r['chunk_id']}\nscore: {r['score']:.3f}\ntext:\n{r['text']}\n"
        )
    return "\n".join(blocks)


def main():
    load_dotenv()  # loads OPENAI_API_KEY from .env
    client = OpenAI()

    index, meta = load_kb()

    print("RAG ready. (Empty ENTER to exit)\n")

    while True:
        q = input("Question: ").strip()
        if not q:
            break

        results = retrieve(q, index, meta, client, k=TOP_K)
        context = build_context(results)

        user_prompt = f"""\
User question:
{q}

SOURCES (use only this):
{context}
"""

        resp = client.responses.create(
            model=LLM_MODEL,
            input=[
                {"role": "system", "content": SYSTEM_RULES},
                {"role": "user", "content": user_prompt},
            ],
        )

        # The simplest way to get text:
        answer = resp.output_text
        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- RETRIEVED SOURCES (top-k) ---")
        for i, r in enumerate(results, start=1):
            print(f"{i}) score={r['score']:.3f} | {Path(r['source_file']).name} - chunk {r['chunk_id']}")
        print()


if __name__ == "__main__":
    main()

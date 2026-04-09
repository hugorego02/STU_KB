from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from kb_config import TOP_K
from kb_retrieval import load_kb, retrieve


def main():
    load_dotenv()
    client = OpenAI()
    index, meta = load_kb()

    while True:
        q = input("\nPergunta (ENTER para sair): ").strip()
        if not q:
            break

        results = retrieve(q, index, meta, client, k=TOP_K)

        print(f"\nTop {len(results)} resultados:")
        for rank, item in enumerate(results, start=1):
            print(f"\n#{rank}  score={item['score']:.3f}")
            print(f"Arquivo: {Path(item['source_file']).name}")
            print(f"Chunk:  {item['chunk_id']}")
            print(f"Trecho:\n{item['text'][:700]}")


if __name__ == "__main__":
    main()

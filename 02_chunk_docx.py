from pathlib import Path
import docx

BASE = Path(__file__).parent
AMOSTRA = BASE / "data_raw" / "_amostra"

# Recommended adjustments for MVP
MAX_CHARS = 1100          # maximum chunk size (characters)
OVERLAP_PARAS = 1         # how many paragraphs to repeat between chunks (1 is great)


def extract_paragraphs_docx(path: Path) -> list[str]:
    """Reads a .docx and returns a list of clean paragraphs (no empty ones)."""
    d = docx.Document(str(path))
    paras = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            # normalize repeated spaces
            t = " ".join(t.split())
            paras.append(t)
    return paras


def chunk_by_paragraphs(paras: list[str], max_chars: int, overlap_paras: int) -> list[str]:
    """
    Joins paragraphs until max_chars. When it exceeds, closes the chunk.
    Overlap: repeats the last overlap_paras paragraphs in the next chunk.
    """
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    i = 0
    while i < len(paras):
        p = paras[i]
        add_len = len(p) + (2 if buf else 0)  # +2 for "\n\n" between paragraphs

        # If it fits, add it
        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
            i += 1
            continue

        # If it doesn't fit and the buffer has content, close the chunk
        if buf:
            chunks.append("\n\n".join(buf).strip())

            # prepare overlap
            if overlap_paras > 0:
                buf = buf[-overlap_paras:]
                buf_len = sum(len(x) for x in buf) + (2 * (len(buf) - 1) if len(buf) > 1 else 0)
            else:
                buf = []
                buf_len = 0
            continue

        # Extreme case: a single paragraph is larger than max_chars
        # We place it alone (later we can improve with sentence-level splitting)
        chunks.append(p)
        i += 1

    # leftover buffer
    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks


def main():
    if not AMOSTRA.exists():
        raise SystemExit(f"I could not find the folder: {AMOSTRA}")

    arquivos = sorted(AMOSTRA.glob("*.docx"))
    if not arquivos:
        raise SystemExit(f"No .docx found in: {AMOSTRA}")

    # take only the first file for inspection (go slowly)
    arq = max(arquivos, key=lambda p: p.stat().st_size)

    print(f"Test file: {arq.name}")

    paras = extract_paragraphs_docx(arq)
    print(f"Extracted paragraphs: {len(paras)}")

    chunks = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)
    print(f"Generated chunks: {len(chunks)}")
    print(f"MAX_CHARS={MAX_CHARS} | OVERLAP_PARAS={OVERLAP_PARAS}\n")

    # show previews of the first chunks
    for idx, ch in enumerate(chunks[:5], start=1):
        preview = ch[:300].replace("\n", " ")
        print(f"--- Chunk {idx} ---")
        print(f"Size: {len(ch)} chars")
        print(f"Preview: {preview}")
        print()

if __name__ == "__main__":
    main()
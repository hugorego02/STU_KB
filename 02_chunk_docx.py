from pathlib import Path
import docx

BASE = Path(__file__).parent
AMOSTRA = BASE / "data_raw" / "_amostra"

# Ajustes recomendados para MVP
MAX_CHARS = 1100          # tamanho máximo do chunk (caracteres)
OVERLAP_PARAS = 1         # quantos parágrafos repetir entre chunks (1 é ótimo)


def extract_paragraphs_docx(path: Path) -> list[str]:
    """Lê um .docx e retorna uma lista de parágrafos limpos (sem vazios)."""
    d = docx.Document(str(path))
    paras = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            # normaliza espaços repetidos
            t = " ".join(t.split())
            paras.append(t)
    return paras


def chunk_by_paragraphs(paras: list[str], max_chars: int, overlap_paras: int) -> list[str]:
    """
    Junta parágrafos até max_chars. Quando estoura, fecha o chunk.
    Overlap: repete os últimos overlap_paras parágrafos no próximo chunk.
    """
    if not paras:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    i = 0
    while i < len(paras):
        p = paras[i]
        add_len = len(p) + (2 if buf else 0)  # +2 por "\n\n" entre parágrafos

        # Se cabe, adiciona
        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
            i += 1
            continue

        # Se não cabe e o buffer tem algo, fecha chunk
        if buf:
            chunks.append("\n\n".join(buf).strip())

            # prepara overlap
            if overlap_paras > 0:
                buf = buf[-overlap_paras:]
                buf_len = sum(len(x) for x in buf) + (2 * (len(buf) - 1) if len(buf) > 1 else 0)
            else:
                buf = []
                buf_len = 0
            continue

        # Caso extremo: um único parágrafo é maior que max_chars
        # A gente coloca ele sozinho (depois podemos melhorar com split por sentença)
        chunks.append(p)
        i += 1

    # sobrou buffer
    if buf:
        chunks.append("\n\n".join(buf).strip())

    return chunks


def main():
    if not AMOSTRA.exists():
        raise SystemExit(f"Não encontrei a pasta: {AMOSTRA}")

    arquivos = sorted(AMOSTRA.glob("*.docx"))
    if not arquivos:
        raise SystemExit(f"Nenhum .docx encontrado em: {AMOSTRA}")

    # pega só o primeiro arquivo para inspeção (ir com calma)
    arq = max(arquivos, key=lambda p: p.stat().st_size)

    print(f"Arquivo de teste: {arq.name}")

    paras = extract_paragraphs_docx(arq)
    print(f"Parágrafos extraídos: {len(paras)}")

    chunks = chunk_by_paragraphs(paras, MAX_CHARS, OVERLAP_PARAS)
    print(f"Chunks gerados: {len(chunks)}")
    print(f"MAX_CHARS={MAX_CHARS} | OVERLAP_PARAS={OVERLAP_PARAS}\n")

    # mostra previews dos primeiros chunks
    for idx, ch in enumerate(chunks[:5], start=1):
        preview = ch[:300].replace("\n", " ")
        print(f"--- Chunk {idx} ---")
        print(f"Tamanho: {len(ch)} chars")
        print(f"Preview: {preview}")
        print()

if __name__ == "__main__":
    main()


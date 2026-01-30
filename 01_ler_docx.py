from pathlib import Path
import docx

BASE = Path(__file__).parent
AMOSTRA = BASE / "data_raw" / "_amostra"

def ler_docx(path: Path) -> str:
    d = docx.Document(str(path))
    textos = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]
    return "\n".join(textos)

def main():
    if not AMOSTRA.exists():
        raise SystemExit(f"I could not find the folder: {AMOSTRA}")

    arquivos = sorted(AMOSTRA.glob("*.docx"))
    if not arquivos:
        raise SystemExit(f"No .docx found in: {AMOSTRA}")

    print(f"I found {len(arquivos)} .docx file(s) in {AMOSTRA}\n")

    for i, arq in enumerate(arquivos, start=1):
        texto = ler_docx(arq)
        preview = texto[:400].replace("\n", " ")
        print(f"[{i}] {arq.name}")
        print(f"    extracted characters: {len(texto)}")
        print(f"    preview: {preview if preview else '(empty)'}\n")

if __name__ == "__main__":
    main()

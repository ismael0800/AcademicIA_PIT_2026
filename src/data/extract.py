"""
Extrator de texto dos PDFs de processos seletivos da UFPI (SIGAA).
Lê a primeira página dos PDFs em data/raw/ e salva o texto limpo em data/processed/.
"""
import sys
import io
import re
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

# Garante saída UTF-8 no terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento do nome da pasta para o nome oficial da categoria
PASTAS_CATEGORIAS = {
    "editais": "edital",
    "homologacoes": "homologacao",
    "recursos": "recurso",
    "resultados": "resultado",
    "avisos": "aviso",
    "retificacoes": "retificacao",
}

def limpar_texto(texto: str) -> str:
    """Normaliza espaços, quebras de linha e remove caracteres de controle."""
    if not texto:
        return ""
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()

def extrair_texto_pdf(pdf_path: Path, max_chars: int = 1000) -> tuple[str, int]:
    """
    Extrai o texto da primeira página do PDF.
    Se a primeira página for muito curta (< 80 caracteres), tenta ler a segunda página.
    """
    try:
        reader = PdfReader(str(pdf_path))
        num_paginas = len(reader.pages)
        if num_paginas == 0:
            return "", 0

        texto_acumulado = ""
        p1 = reader.pages[0].extract_text() or ""
        texto_acumulado += p1

        if len(limpar_texto(texto_acumulado)) < 80 and num_paginas > 1:
            p2 = reader.pages[1].extract_text() or ""
            texto_acumulado += " " + p2

        texto_limpo = limpar_texto(texto_acumulado)
        return texto_limpo[:max_chars], num_paginas
    except Exception as e:
        print(f"[!] Erro ao ler {pdf_path.name}: {e}")
        return "", 0

def processar_todos():
    registros = []
    print("[*] Iniciando extracao dos PDFs da UFPI (SIGAA)...")
    print("-" * 55)

    for pasta_nome, cat in PASTAS_CATEGORIAS.items():
        pasta_cat = RAW_DIR / pasta_nome
        if not pasta_cat.exists():
            print(f"[!] Pasta {pasta_nome} nao encontrada em {RAW_DIR}")
            continue

        pdfs = list(pasta_cat.glob("*.pdf"))
        print(f"[-] Pasta: {pasta_nome:<14} -> Categoria: {cat:<12} | {len(pdfs)} PDFs")

        for pdf in pdfs:
            texto, num_pags = extrair_texto_pdf(pdf)
            registros.append({
                "arquivo": pdf.name,
                "categoria": cat,
                "texto": texto,
                "num_caracteres": len(texto),
                "num_paginas": num_pags,
                "fonte": "SIGAA/UFPI - Processos Seletivos Stricto Sensu"
            })

    df = pd.DataFrame(registros)
    total = len(df)
    vazios = (df["num_caracteres"] == 0).sum() if total > 0 else 0

    print("-" * 55)
    print(f"[OK] Total de arquivos processados: {total}")
    print(f"[OK] Arquivos com texto extraido:   {total - vazios}")
    if vazios > 0:
        print(f"[!] Arquivos sem texto (possivel imagem escaneada): {vazios}")

    parquet_path = PROCESSED_DIR / "documentos.parquet"
    csv_path = PROCESSED_DIR / "documentos.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Dados salvos em:")
    print(f"     • {parquet_path}")
    print(f"     • {csv_path}")
    return df

if __name__ == "__main__":
    processar_todos()

"""
Validador de qualidade e esquema do dataset processado (SIGAA/UFPI).
Verifica integridade, textos vazios, duplicatas e distribuição das classes.
"""
import sys
import io
from pathlib import Path
import pandas as pd

# Garante saída UTF-8 no terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROCESSED_FILE = Path("data/processed/documentos.parquet")

def validar_dataset():
    print("[*] Executando validacao de qualidade dos dados...")
    print("-" * 55)

    assert PROCESSED_FILE.exists(), f"Arquivo {PROCESSED_FILE} nao existe. Execute extract.py primeiro."
    df = pd.read_parquet(PROCESSED_FILE)

    # 1. Validação de presença das colunas obrigatórias
    colunas_obrigatorias = {"arquivo", "categoria", "texto", "num_caracteres", "fonte"}
    faltantes = colunas_obrigatorias - set(df.columns)
    assert not faltantes, f"[ERRO] Colunas ausentes no dataset: {faltantes}"
    print("[OK] 1. Esquema de colunas valido e presente.")

    # 2. Total de linhas
    total = len(df)
    assert total > 0, "[ERRO] O dataset esta completamente vazio!"
    print(f"[OK] 2. Total de registros analisados: {total} documentos.")

    # 3. Textos vazios ou muito curtos (< 50 caracteres)
    vazios = df[df["num_caracteres"] < 50]
    if len(vazios) > 0:
        print(f"[!] 3. Atencao: {len(vazios)} documento(s) com texto muito curto (< 50 caracteres):")
        for _, row in vazios.iterrows():
            print(f"      • [{row['categoria']}] {row['arquivo']} ({row['num_caracteres']} caracteres)")
    else:
        print("[OK] 3. Todos os documentos possuem texto suficiente extraido.")

    # 4. Checagem de textos duplicados
    duplicados = df.duplicated(subset=["texto"]).sum()
    print(f"[i]  4. Textos com repeticao exata encontrados: {duplicados}")

    # 5. Distribuição e balanceamento das 6 categorias
    print("-" * 55)
    print("[*] 5. Distribuicao por Categoria:")
    contagem = df["categoria"].value_counts()
    for cat, qtd in contagem.items():
        pct = (qtd / total) * 100
        print(f"   • {cat:<14}: {qtd:>2} arquivos ({pct:>5.1f}%)")

    assert len(contagem) >= 2, "[ERRO] A tarefa precisa de no minimo 2 classes diferentes!"
    print("-" * 55)
    print("[OK] Validacao concluida com sucesso! Os dados estao prontos para a separacao (split).")
    return True

if __name__ == "__main__":
    validar_dataset()

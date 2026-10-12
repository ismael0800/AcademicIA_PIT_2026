"""
Divisão estratificada dos dados limpos em Treino, Validação e Teste Reservado.
Gera os arquivos finais na camada data/curated/ sem vazamento de dados (data leakage).
"""
import sys
import io
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Garante saída UTF-8 no terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROCESSED_FILE = Path("data/processed/documentos.parquet")
CURATED_DIR = Path("data/curated")
CURATED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

def realizar_split():
    print("[*] Iniciando divisao estratificada dos dados (Split)...")
    print("-" * 55)

    assert PROCESSED_FILE.exists(), f"Arquivo {PROCESSED_FILE} nao existe. Execute extract.py primeiro."
    df = pd.read_parquet(PROCESSED_FILE)

    # 1. Filtra apenas documentos com texto valido (descarta PDFs escaneados com 0 caracteres)
    df_valido = df[df["num_caracteres"] >= 50].copy()
    total_validos = len(df_valido)
    descartados = len(df) - total_validos

    print(f"[OK] Total de documentos carregados: {len(df)}")
    print(f"[OK] Descartados por falta de texto (fotos escaneadas): {descartados}")
    print(f"[OK] Documentos validos selecionados para o split: {total_validos}")
    print("-" * 55)

    # 2. Split Estratificado: 70% Treino, 30% Temporário (que vira 15% Validação e 15% Teste)
    # Stratify garante que todas as 6 classes mantenham a mesma proporção em todos os conjuntos
    train, temp = train_test_split(
        df_valido,
        test_size=0.30,
        random_state=SEED,
        stratify=df_valido["categoria"]
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        stratify=temp["categoria"]
    )

    # 3. Salva os arquivos finais em Parquet e CSV para facilitar visualização
    train.to_parquet(CURATED_DIR / "train.parquet", index=False)
    val.to_parquet(CURATED_DIR / "validation.parquet", index=False)
    test.to_parquet(CURATED_DIR / "test.parquet", index=False)

    train.to_csv(CURATED_DIR / "train.csv", index=False, encoding="utf-8-sig")
    val.to_csv(CURATED_DIR / "validation.csv", index=False, encoding="utf-8-sig")
    test.to_csv(CURATED_DIR / "test.csv", index=False, encoding="utf-8-sig")

    # 4. Relatório de distribuição
    print("[*] Resumo dos Splits:")
    print(f"   • Treino      (70%): {len(train):>2} documentos -> data/curated/train.parquet")
    print(f"   • Validacao   (15%): {len(val):>2} documentos -> data/curated/validation.parquet")
    print(f"   • Teste Final (15%): {len(test):>2} documentos -> data/curated/test.parquet (CONGELADO)")
    print("-" * 55)

    print("[*] Proporcao de classes no Treino:")
    for cat, qtd in train["categoria"].value_counts().items():
        print(f"   • {cat:<14}: {qtd:>2} documentos")

    print("-" * 55)
    print("[OK] Split concluido com sucesso e sem vazamento de dados!")
    return train, val, test

if __name__ == "__main__":
    realizar_split()

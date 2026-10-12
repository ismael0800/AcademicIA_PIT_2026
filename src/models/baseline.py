"""
Modelo Baseline de Classificação de Documentos (SIGAA/UFPI).
Utiliza TF-IDF clássico + Regressão Logística para estabelecer a linha de base
que a rede neural em PyTorch precisará superar.
Avaliado estritamente no conjunto de VALIDAÇÃO (teste reservado permanece congelado).
"""
import sys
import io
import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Garante saída UTF-8 no terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path("data/curated")
REPORTS_DIR = Path("reports/metrics")
MODELS_DIR = Path("models")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def treinar_baseline():
    print("[*] Iniciando treinamento e avaliacao do modelo BASELINE...")
    print("-" * 60)

    # 1. Carrega Treino e Validação (NUNCA carrega test.parquet aqui)
    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "validation.parquet"

    assert train_path.exists(), f"Arquivo {train_path} nao encontrado."
    assert val_path.exists(), f"Arquivo {val_path} nao encontrado."

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    X_train, y_train = df_train["texto"], df_train["categoria"]
    X_val, y_val = df_val["texto"], df_val["categoria"]

    print(f"[OK] Dados de Treino:     {len(X_train)} documentos")
    print(f"[OK] Dados de Validacao:  {len(X_val)} documentos")
    print(f"[OK] Conjunto de Teste:   CONGELADO (nao utilizado no baseline)")
    print("-" * 60)

    # 2. Pipeline Clássico: TF-IDF (unigramas e bigramas) + Regressão Logística
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=500,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=200,
            random_state=42
        ))
    ])

    # 3. Treinamento
    pipeline.fit(X_train, y_train)

    # 4. Avaliação na Validação
    y_pred = pipeline.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print("[*] RESULTADOS DO BASELINE NA VALIDACAO:")
    print(f"   • Acuracia:         {acc * 100:.2f}%")
    print(f"   • F1-Score Macro:   {f1_macro:.4f}")
    print(f"   • F1-Score Ponderado: {f1_weighted:.4f}")
    print("-" * 60)

    print("[*] Relatorio Detalhado por Categoria:")
    print(classification_report(y_val, y_pred, zero_division=0))

    # 5. Salva métricas em JSON para auditoria e MLOps
    metricas = {
        "modelo": "Baseline TF-IDF + LogisticRegression",
        "dataset_train_rows": len(df_train),
        "dataset_val_rows": len(df_val),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "classes": sorted(list(df_train["categoria"].unique()))
    }

    metrics_file = REPORTS_DIR / "baseline.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # 6. Salva modelo treinado
    model_file = MODELS_DIR / "baseline.joblib"
    joblib.dump(pipeline, model_file)

    print(f"[OK] Metricas salvas em: {metrics_file}")
    print(f"[OK] Modelo salvo em:    {model_file}")
    print("-" * 60)
    print("🎯 Referencia definida! Esta eh a nota que a rede neural em PyTorch precisara superar.")
    return metricas

if __name__ == "__main__":
    treinar_baseline()

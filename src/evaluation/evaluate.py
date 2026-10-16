"""
Script de Avaliação Final, Matriz de Confusão e Análise de Erros (AcademicIA).
Avalia o melhor modelo (best.pt) no conjunto de validação e no TESTE RESERVADO.
Gera imagem da Matriz de Confusão e relatório qualitativo de erros.
"""
import sys
import io
import json
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from src.models.network import TextClassifier
from src.data.dataset import Vocabulary, EditaisDataset, LABEL_TO_ID, ID_TO_LABEL
from torch.utils.data import DataLoader

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DATA_DIR = Path("data/curated")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def avaliar_conjunto(model, dataset, classes_nomes):
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    y_true = []
    y_pred = []
    probs_list = []

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            y_true.extend(y_batch.tolist())
            y_pred.extend(preds.tolist())
            probs_list.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(probs_list)


def plotar_matriz_confusao(y_true, y_pred, classes_nomes, output_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Plota mapa de calor com anotação numérica
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matriz de Confusão — Teste Reservado (AcademicIA)", fontsize=13, pad=12)
    plt.colorbar()

    tick_marks = np.arange(len(classes_nomes))
    plt.xticks(tick_marks, classes_nomes, rotation=35, ha="right", fontsize=10)
    plt.yticks(tick_marks, classes_nomes, fontsize=10)

    # Escreve os números em cada célula
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.ylabel("Classe Real (Gabarito)", fontsize=11)
    plt.xlabel("Classe Predita pela IA", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def executar_avaliacao_completa():
    print("=" * 65)
    print("[*] INICIANDO AVALIAÇÃO DEFINITIVA DO MODELO (best.pt)")
    print("=" * 65)

    # 1. Carrega o vocabulário e o melhor checkpoint
    vocab_path = MODELS_DIR / "vocab.json"
    checkpoint_path = MODELS_DIR / "best.pt"

    assert vocab_path.exists(), f"Vocabulário não encontrado em {vocab_path}"
    assert checkpoint_path.exists(), f"Checkpoint best.pt não encontrado em {checkpoint_path}. Execute o treino primeiro."

    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Recria o modelo e carrega os pesos
    classes_nomes = [ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    model = TextClassifier(
        vocab_size=checkpoint.get("vocab_size", len(vocab.w2i)),
        embedding_dim=checkpoint.get("embedding_dim", 64),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_classes=len(classes_nomes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[OK] Modelo restaurado com sucesso! Salvo originalmente na Época {checkpoint['epoch']} com Val Loss de {checkpoint['val_loss']:.4f}")
    print("-" * 65)

    # 3. Carrega o conjunto de Validação e de Teste Reservado
    df_val = pd.read_parquet(DATA_DIR / "validation.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test.parquet")

    val_ds = EditaisDataset(df_val, vocab, max_len=128)
    test_ds = EditaisDataset(df_test, vocab, max_len=128)

    # 4. Avaliação na Validação
    y_val_true, y_val_pred, _ = avaliar_conjunto(model, val_ds, classes_nomes)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

    # 5. Avaliação no TESTE RESERVADO (Abertura oficial)
    y_test_true, y_test_pred, test_probs = avaliar_conjunto(model, test_ds, classes_nomes)
    test_acc = accuracy_score(y_test_true, y_test_pred)
    test_f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    test_f1_weighted = f1_score(y_test_true, y_test_pred, average="weighted", zero_division=0)

    print("📊 RESULTADOS NO TESTE RESERVADO (11 documentos nunca vistos):")
    print(f"   • Acurácia no Teste:       {test_acc * 100:.2f}%")
    print(f"   • F1-Score Macro:          {test_f1:.4f}")
    print(f"   • F1-Score Ponderado:      {test_f1_weighted:.4f}")
    print("-" * 65)

    # 6. Carrega o Baseline anterior para comparar lado a lado
    baseline_path = METRICS_DIR / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        print("🏆 COMPARAÇÃO: BASELINE vs. REDE NEURAL (Validação):")
        print(f"   • Baseline (TF-IDF + LogReg): Acurácia = {base_data['accuracy']*100:.2f}% | F1-Macro = {base_data['f1_macro']:.4f}")
        print(f"   • Rede Neural (PyTorch):      Acurácia = {val_acc*100:.2f}% | F1-Macro = {val_f1:.4f}")
        print("-" * 65)

    # 7. Relatório Detalhado de Classificação no Teste
    print("[*] Relatório Detalhado por Classe no Teste:")
    relatorio = classification_report(
        y_test_true,
        y_test_pred,
        target_names=classes_nomes,
        zero_division=0,
        output_dict=False
    )
    print(relatorio)

    # 8. Gera a Matriz de Confusão em Imagem
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    plotar_matriz_confusao(y_test_true, y_test_pred, classes_nomes, cm_path)
    print(f"[OK] Imagem da Matriz de Confusão salva em: {cm_path}")

    # 9. Análise Qualitativa de Erros
    erros = []
    for idx, (real, pred) in enumerate(zip(y_test_true, y_test_pred)):
        if real != pred:
            doc_info = df_test.iloc[idx]
            erros.append({
                "arquivo": doc_info["arquivo"],
                "classe_real": ID_TO_LABEL[real],
                "classe_predita": ID_TO_LABEL[pred],
                "confianca_predita": float(test_probs[idx][pred]),
                "trecho_texto": doc_info["texto"][:250] + "...",
                "hipotese_do_erro": "Vocabulário com termos de cabeçalho compartilhados ou presença de expressões de ambas as categorias."
            })

    error_analysis_path = METRICS_DIR / "error_analysis.json"
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        json.dump(erros, f, indent=2, ensure_ascii=False)

    print("-" * 65)
    print(f"🔍 ANÁLISE QUALITATIVA DE ERROS ({len(erros)} erros identificados no Teste):")
    if erros:
        for e in erros:
            print(f" • Arquivo: {e['arquivo']}")
            print(f"   Esperado: [{e['classe_real']}] | IA chutou: [{e['classe_predita']}] (Confiança: {e['confianca_predita']*100:.1f}%)")
            print(f"   Trecho: \"{e['trecho_texto'][:100]}...\"")
    else:
        print(" 🎉 Nenhum erro no teste reservado!")

    print(f"[OK] Relatório completo de erros salvo em: {error_analysis_path}")
    print("=" * 65)

    # 10. Salva métricas finais consolidadas
    metrics_final = {
        "modelo": "TextClassifier (PyTorch)",
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_f1_weighted": float(test_f1_weighted),
        "val_accuracy": float(val_acc),
        "val_f1_macro": float(val_f1),
        "total_erros_teste": len(erros)
    }
    with open(METRICS_DIR / "test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics_final, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    executar_avaliacao_completa()

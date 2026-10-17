"""
Script de Treinamento em PyTorch com MLflow, Checkpoints e Early Stopping (AcademicIA).
Treina a rede híbrida TextClassifier (Conv1D + Self-Attention Pooling) nos dados do Piauí.
"""
import sys
import io
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import mlflow

from src.models.network import TextClassifier
from src.data.dataset import criar_dataloaders, fixar_seed

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPORTS_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")
MODELS_DIR = Path("models")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def treinar_experimento(
    lr: float = 0.002,
    batch_size: int = 8,
    embedding_dim: int = 128,
    hidden_dim: int = 96,
    num_filters: int = 48,
    epochs: int = 35,
    patience: int = 12,
    exp_name: str = "AcademicIA_Treinamento",
    run_name: str = "run_padrao"
):
    fixar_seed(42)
    device = torch.device("cpu")

    print(f"\n{'='*65}")
    print(f"[*] INICIANDO EXPERIMENTO: {run_name} (lr={lr}, emb={embedding_dim}, filters={num_filters})")
    print(f"{'='*65}")

    # 1. Carrega os DataLoaders (com data augmentation e vocabulário semântico)
    train_loader, val_loader, test_loader, vocab = criar_dataloaders(batch_size=batch_size, max_len=200)
    vocab_size = len(vocab.w2i)
    num_classes = 6

    # 2. Instancia a arquitetura híbrida
    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_prob=0.25,
        num_filters=num_filters
    ).to(device)

    # 3. Loss com label smoothing e Otimizador AdamW com Cosine Annealing
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    melhor_epoca = 0
    epocas_sem_melhora = 0

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_filters": num_filters,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "architecture": "TextCNN + Self-Attention Pooling",
            "optimizer": "AdamW + CosineAnnealing",
            "seed": 42
        })

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_total = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_total += loss.item() * len(y_batch)

            scheduler.step()
            train_loss = train_loss_total / len(train_loader.dataset)

            # Fase de Validação
            model.eval()
            val_loss_total = 0.0
            val_preds = []
            val_targets = []

            with torch.inference_mode():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = loss_fn(logits, y_batch)
                    val_loss_total += loss.item() * len(y_batch)
                    
                    preds = logits.argmax(dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(y_batch.cpu().numpy())

            val_loss = val_loss_total / len(val_loader.dataset)
            val_acc = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1_macro", val_f1, step=epoch)

            if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
                print(f"Época {epoch:>2}/{epochs} | Treino Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:>5.1f}% | Val F1: {val_f1:.4f}")

            # Salva melhor checkpoint baseado em val_loss ou val_acc
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                melhor_epoca = epoch
                epocas_sem_melhora = 0
                
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_filters": num_filters,
                    "num_classes": num_classes
                }
                
                torch.save(checkpoint_data, MODELS_DIR / f"best_{run_name}.pt")
                if run_name == "run_padrao":
                    torch.save(checkpoint_data, MODELS_DIR / "best.pt")
                
                print(f"   [+] Novo melhor modelo salvo! (Época {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%)")
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= patience:
                    print(f"\n[!] Early Stopping ativado na época {epoch} (sem melhora há {patience} épocas).")
                    break

        mlflow.log_artifact(str(MODELS_DIR / f"best_{run_name}.pt"))
        print(f"\n[OK] Treinamento '{run_name}' concluído! Melhor época: {melhor_epoca} (Val Loss: {best_val_loss:.4f})")
        return history


def gerar_grafico_curvas(history, filename="training_curves.png"):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epocas = range(1, len(history["train_loss"]) + 1)

    # 1. Gráfico de Perda (Loss)
    ax1.plot(epocas, history["train_loss"], label="Perda no Treino (Train Loss)", color="#2563eb", lw=2, marker="o", markersize=3)
    ax1.plot(epocas, history["val_loss"], label="Perda na Validação (Val Loss)", color="#dc2626", lw=2, marker="s", markersize=3)
    ax1.set_title("Evolução da Função de Perda (CrossEntropy)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Épocas", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Gráfico de Acurácia e F1-Score
    ax2.plot(epocas, [a * 100 for a in history["val_acc"]], label="Acurácia de Validação (%)", color="#16a34a", lw=2, marker="^", markersize=3)
    ax2.plot(epocas, [f * 100 for f in history["val_f1"]], label="F1-Score Macro (%)", color="#9333ea", lw=2, marker="d", markersize=3)
    ax2.set_title("Evolução da Acurácia e F1-Score (Validação)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Épocas", fontsize=11)
    ax2.set_ylabel("Percentual (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = REPORTS_DIR / filename
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Curvas de treinamento salvas em: {output_path}")


def main():
    print("=" * 65)
    print("🚀 PIPELINE DE TREINAMENTO REFINADO - AcademicIA")
    print("=" * 65)

    hist_padrao = treinar_experimento(
        lr=0.002,
        batch_size=8,
        embedding_dim=128,
        hidden_dim=96,
        num_filters=48,
        epochs=35,
        patience=12,
        run_name="run_padrao"
    )

    gerar_grafico_curvas(hist_padrao, "training_curves.png")
    print("\n🎉 Treinamento refinado concluído com sucesso!")


if __name__ == "__main__":
    main()

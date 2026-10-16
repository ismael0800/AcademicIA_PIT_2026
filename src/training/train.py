"""
Script de Treinamento Completo em PyTorch com MLflow, Checkpoints e Early Stopping (AcademicIA).
Treina a rede TextClassifier nos dados da UFPI, monitora a validação e salva o modelo best.pt.
"""
import sys
import io
import json
from pathlib import Path

# Garante que a raiz do projeto esteja no sys.path
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
    lr: float = 0.001,
    batch_size: int = 8,
    embedding_dim: int = 64,
    hidden_dim: int = 64,
    epochs: int = 25,
    patience: int = 6,
    exp_name: str = "AcademicIA_Treinamento",
    run_name: str = "run_padrao"
):
    fixar_seed(42)
    device = torch.device("cpu")

    print(f"\n{'='*65}")
    print(f"[*] INICIANDO EXPERIMENTO: {run_name} (lr={lr}, emb={embedding_dim})")
    print(f"{'='*65}")

    # 1. Carrega os DataLoaders (Treino e Validação)
    train_loader, val_loader, test_loader, vocab = criar_dataloaders(batch_size=batch_size)
    vocab_size = len(vocab.w2i)
    num_classes = 6

    # 2. Instancia o modelo
    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_prob=0.2
    ).to(device)

    # 3. Loss (CrossEntropy) e Otimizador (AdamW)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Histórico de métricas
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    melhor_epoca = 0
    epocas_sem_melhora = 0

    # Configuração do MLflow
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "optimizer": "AdamW",
            "loss_function": "CrossEntropyLoss",
            "seed": 42
        })

        for epoch in range(1, epochs + 1):
            # Fase de Treino
            model.train()
            train_loss_total = 0.0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                
                # Clipping de gradiente para segurança numérica
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss_total += loss.item() * len(y_batch)

            train_loss = train_loss_total / len(train_loader.dataset)

            # Fase de Validação (Inference Mode sem gradientes)
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

            # Registra no MLflow a cada época
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_f1_macro", val_f1, step=epoch)

            # Impressão do progresso
            if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
                print(f"Época {epoch:>2}/{epochs} | Treino Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:>5.1f}% | Val F1: {val_f1:.4f}")

            # Lógica de Checkpoint e Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                melhor_epoca = epoch
                epocas_sem_melhora = 0
                
                # Salva os pesos do melhor modelo
                checkpoint_path = MODELS_DIR / f"best_{run_name}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_classes": num_classes
                }, checkpoint_path)
                
                # Se for a run padrão, salva também como o best.pt principal
                if run_name == "run_padrao":
                    torch.save(torch.load(checkpoint_path), MODELS_DIR / "best.pt")
            else:
                epocas_sem_melhora += 1
                if epocas_sem_melhora >= patience:
                    print(f"[*] Early Stopping acionado na época {epoch}! (Sem melhora por {patience} épocas)")
                    break

        print(f"[OK] Melhor Checkpoint salvo na Época {melhor_epoca} com Val Loss: {best_val_loss:.4f}")
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", melhor_epoca)

        # 4. Gera e salva o gráfico das curvas de aprendizado
        fig_path = REPORTS_DIR / f"curvas_{run_name}.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["train_loss"], label="Treino (Train Loss)", color="#2563eb", linewidth=2)
        ax1.plot(history["val_loss"], label="Validação (Val Loss)", color="#dc2626", linewidth=2)
        ax1.set_title("Curva de Perda (CrossEntropy Loss)")
        ax1.set_xlabel("Épocas")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        ax2.plot([acc * 100 for acc in history["val_acc"]], label="Acurácia (%)", color="#16a34a", linewidth=2)
        ax2.plot([f1 * 100 for f1 in history["val_f1"]], label="F1-Score Macro (%)", color="#9333ea", linewidth=2)
        ax2.set_title("Evolução das Métricas na Validação")
        ax2.set_xlabel("Épocas")
        ax2.set_ylabel("Porcentagem (%)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # Salva o gráfico também como training_curves.png principal
        if run_name == "run_padrao":
            plt.figure()
            import shutil
            shutil.copy(fig_path, REPORTS_DIR / "training_curves.png")

        mlflow.log_artifact(str(fig_path))
        print(f"[OK] Gráfico das curvas salvo em: {fig_path}")

    return {
        "run_name": run_name,
        "best_epoch": melhor_epoca,
        "best_val_loss": best_val_loss,
        "val_acc": max(history["val_acc"]),
        "val_f1": max(history["val_f1"])
    }


def executar_bateria_experimentos():
    print("[*] INICIANDO EXPERIMENTOS COMPARATIVOS NO MLFLOW...")
    
    # Run 1: Configuração Padrão (lr=0.001, emb=64)
    res1 = treinar_experimento(
        lr=0.001,
        embedding_dim=64,
        run_name="run_padrao",
        epochs=25
    )

    # Run 2: Hipótese de Taxa de Aprendizado Alta (lr=0.01)
    res2 = treinar_experimento(
        lr=0.01,
        embedding_dim=64,
        run_name="run_lr_alto",
        epochs=20
    )

    # Run 3: Hipótese de Maior Capacidade Semântica (emb=128)
    res3 = treinar_experimento(
        lr=0.001,
        embedding_dim=128,
        run_name="run_emb_128",
        epochs=25
    )

    print("\n" + "="*65)
    print("📊 RESUMO DOS 3 EXPERIMENTOS NO MLFLOW:")
    print("="*65)
    for r in [res1, res2, res3]:
        print(f" • {r['run_name']:<15} | Melhor Época: {r['best_epoch']:>2} | Val Loss: {r['best_val_loss']:.4f} | Acurácia Máx: {r['val_acc']*100:.1f}%")
    print("="*65)


if __name__ == "__main__":
    executar_bateria_experimentos()

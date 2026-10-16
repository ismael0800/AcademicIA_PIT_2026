"""
Arquitetura da Rede Neural em PyTorch para Classificação de Editais (AcademicIA).
Modelo: TextClassifier (Embeddings próprios + Average Pooling + Camadas Densas + ReLU).
Pesos 100% inicializados aleatoriamente e treinados do zero (CPU-first).
"""
import sys
import io
import torch
import torch.nn as nn

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

class TextClassifier(nn.Module):
    """
    Rede Neural para classificação de texto em 6 categorias.
    
    Camadas:
      1. Embedding: mapeia IDs de palavras em vetores contínuos de dimensão D.
      2. Pooling: calcula a média dos vetores do documento (Deep Averaging Network).
      3. Camada Oculta: Linear(D, hidden_dim) + ativação ReLU + Dropout.
      4. Camada de Saída: Linear(hidden_dim, num_classes) gerando os 6 logits.
    """
    def __init__(
        self,
        vocab_size: int = 2500,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 6,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        
        # 1. Camada de Embeddings treináveis (padding_idx=0 para ignorar preenchimento)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 2. Camadas Densas (MLP) com não-linearidade e regularização
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Fluxo Forward da rede:
          Entrada:  tensor com shape [Batch, Comprimento_do_Texto]
          Saída:    tensor com logits [Batch, 6_Classes]
        """
        # [Batch, Seq_Len] -> [Batch, Seq_Len, Embedding_Dim]
        embedded = self.embedding(token_ids)
        
        # Pooling: Média temporal dos embeddings das palavras -> [Batch, Embedding_Dim]
        # (cria uma representação vetorial compacta do documento inteiro)
        pooled = embedded.mean(dim=1)
        
        # Passa pelas camadas lineares e ativação -> [Batch, Num_Classes]
        logits = self.classifier(pooled)
        return logits


def demonstrar_ciclo_treinamento():
    print("[*] Iniciando demonstracao da Rede Neural e Backpropagation...")
    print("=" * 60)

    # 1. Instanciação do modelo com as 6 classes do projeto
    vocab_size = 2500
    embedding_dim = 64
    hidden_dim = 64
    num_classes = 6

    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_prob=0.2
    )

    # 2. Contagem de parâmetros treináveis
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 1. Total de parâmetros treináveis: {total_params:,}")
    print(f"        • Embedding ({vocab_size}x{embedding_dim}): {vocab_size * embedding_dim:,} parâmetros")
    print(f"        • Camada Oculta ({embedding_dim}x{hidden_dim} + bias): {embedding_dim * hidden_dim + hidden_dim:,} parâmetros")
    print(f"        • Camada de Saída ({hidden_dim}x{num_classes} + bias): {hidden_dim * num_classes + num_classes:,} parâmetros")
    print("-" * 60)

    # 3. Shape dos Tensores de Entrada e Saída
    batch_size = 4
    seq_len = 128  # exemplo com 128 palavras por documento
    dummy_input = torch.randint(1, vocab_size, (batch_size, seq_len))
    dummy_target = torch.tensor([0, 2, 4, 1], dtype=torch.long)  # classes reais

    print(f"[OK] 2. Formato (Shape) dos Tensores:")
    print(f"        • Entrada (x): [Batch={batch_size}, Sequencia={seq_len}] -> IDs das palavras")
    print(f"        • Alvo    (y): [Batch={batch_size}] -> Classe correta de cada documento")
    print("-" * 60)

    # 4. Demonstração do ciclo: Forward -> Loss -> Backward -> Optimizer Step
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    print("[*] 3. Executando o Ciclo de Aprendizado (1 Step):")

    # Passo A: Forward (Previsão)
    optimizer.zero_grad()
    logits = model(dummy_input)
    print(f"   [A] FORWARD: Logits gerados com shape {list(logits.shape)} (4 exemplos x 6 classes)")

    # Passo B: Loss (Cálculo do Erro)
    loss = loss_fn(logits, dummy_target)
    print(f"   [B] LOSS (CrossEntropy): {loss.item():.4f}")

    # Passo C: Backward (Cálculo dos Gradientes via Backpropagation)
    loss.backward()
    print(f"   [C] BACKWARD: Gradientes calculados via regra da cadeia!")

    # Inspeciona as normas dos gradientes para provar que a rede está aprendendo
    print("\n[*] 4. Inspecao dos Gradientes (Norma L2 de cada camada):")
    for name, param in model.named_parameters():
        if param.grad is not None:
            norma = param.grad.norm().item()
            print(f"   • {name:<30} -> Norma do Gradiente = {norma:.6f}")

    # Passo D: Atualização dos Pesos
    optimizer.step()
    print(f"\n   [D] OPTIMIZER STEP: Pesos atualizados com sucesso pelo AdamW!")
    print("=" * 60)
    print("🎉 Prova de Deep Learning completa! O fluxo forward/backward esta 100% operacional.")


if __name__ == "__main__":
    demonstrar_ciclo_treinamento()

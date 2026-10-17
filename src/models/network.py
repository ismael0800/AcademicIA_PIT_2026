"""
Arquitetura da Rede Neural em PyTorch para Classificação de Editais (AcademicIA).
Modelo: TextClassifier Híbrido (Multi-Scale Conv1D + Self-Attention Pooling + MLP).
Pesos 100% inicializados aleatoriamente e treinados do zero (CPU-first).
"""
import sys
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class TextClassifier(nn.Module):
    """
    Rede Neural Híbrida para classificação de documentos acadêmicos em 6 categorias.
    
    Camadas:
      1. Embedding: mapeia IDs de palavras em vetores densos contínuos de dimensão D.
      2. Multi-Scale Conv1D (TextCNN): filtros de convolução de tamanho 2, 3 e 4
         que capturam n-gramas essenciais (ex: "homologação das inscrições", "edital de abertura").
         Seguidos por Max-Pooling temporal (extrai o sinal n-grama mais forte independente da posição).
      3. Self-Attention Pooling: mecanismo de atenção neural que pondera a relevância contextual
         das palavras do texto, reduzindo o peso de conectivos e cabeçalhos neutros.
      4. Camadas Densas (MLP) com LayerNorm, ReLU, Dropout e projeção para 6 logits.
    """
    def __init__(
        self,
        vocab_size: int = 2500,
        embedding_dim: int = 128,
        hidden_dim: int = 96,
        num_classes: int = 6,
        dropout_prob: float = 0.25,
        num_filters: int = 48
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_filters = num_filters
        
        # 1. Camada de Embeddings treináveis (padding_idx=0 para ignorar preenchimento)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 2. Convoluções 1D multi-escala para captura de bigramas, trigramas e 4-gramas
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(embedding_dim, num_filters, kernel_size=4)
        
        # 3. Mecanismo de Self-Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Camadas Densas (MLP) com normalização e regularização
        total_features = (num_filters * 3) + embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
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
        # Garante comprimento mínimo para conv4 (kernel_size=4)
        if token_ids.size(1) < 4:
            pad_len = 4 - token_ids.size(1)
            token_ids = F.pad(token_ids, (0, pad_len), value=0)

        # [Batch, Seq_Len] -> [Batch, Seq_Len, Embedding_Dim]
        embedded = self.embedding(token_ids)
        
        # 1. Self-Attention Pooling (Média Ponderada Neural)
        att_weights = F.softmax(self.attention(embedded), dim=1) # [Batch, Seq_Len, 1]
        att_pooled = (embedded * att_weights).sum(dim=1)         # [Batch, Embedding_Dim]
        
        # 2. Multi-Scale Conv1D + Global Max-Pooling
        emb_transposed = embedded.transpose(1, 2) # [Batch, Embedding_Dim, Seq_Len]
        c2 = F.relu(self.conv2(emb_transposed)).max(dim=2)[0] # [Batch, num_filters]
        c3 = F.relu(self.conv3(emb_transposed)).max(dim=2)[0] # [Batch, num_filters]
        c4 = F.relu(self.conv4(emb_transposed)).max(dim=2)[0] # [Batch, num_filters]
        conv_pooled = torch.cat([c2, c3, c4], dim=1)          # [Batch, num_filters * 3]
        
        # 3. Fusão de Features Locais (N-grams) + Globais (Atenção)
        features = torch.cat([conv_pooled, att_pooled], dim=1)
        logits = self.classifier(features)
        return logits


def demonstrar_ciclo_treinamento():
    print("[*] Iniciando demonstração da Rede Neural Híbrida (Conv1D + Attention)...")
    print("=" * 65)

    vocab_size = 1500
    embedding_dim = 128
    hidden_dim = 96
    num_classes = 6

    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout_prob=0.25,
        num_filters=48
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 1. Total de parâmetros treináveis: {total_params:,}")
    print(f"        • Embeddings: {vocab_size * embedding_dim:,} parâmetros")
    print(f"        • Convoluções (2, 3, 4-grams): {sum(p.numel() for m in [model.conv2, model.conv3, model.conv4] for p in m.parameters()):,} parâmetros")
    print(f"        • Self-Attention Pooling: {sum(p.numel() for p in model.attention.parameters()):,} parâmetros")
    print(f"        • MLP Classificador: {sum(p.numel() for p in model.classifier.parameters()):,} parâmetros")
    print("-" * 65)

    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(1, vocab_size, (batch_size, seq_len))
    dummy_target = torch.tensor([0, 1, 3, 5], dtype=torch.long)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)

    optimizer.zero_grad()
    logits = model(dummy_input)
    print(f"[OK] 2. Logits gerados com shape {list(logits.shape)} (Batch={batch_size}, Classes={num_classes})")

    loss = loss_fn(logits, dummy_target)
    print(f"[OK] 3. Loss (CrossEntropy): {loss.item():.4f}")

    loss.backward()
    print("[OK] 4. Backpropagation executado e gradientes calculados.")

    optimizer.step()
    print("[OK] 5. Optimizer step executado com sucesso!")
    print("=" * 65)
    print("🎉 Rede Neural Híbrida 100% operacional!")


if __name__ == "__main__":
    demonstrar_ciclo_treinamento()

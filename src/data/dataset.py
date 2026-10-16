"""
Pipeline de Dados em PyTorch: Dataset, Vocabulário e DataLoaders (AcademicIA).
Transforma textos limpos em tensores com controle determinístico de seed e batch.
"""
import sys
import io
import json
import re
import random
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CURATED_DIR = Path("data/curated")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Mapeamento oficial das 6 classes para IDs numéricos
LABEL_TO_ID = {
    "edital": 0,
    "homologacao": 1,
    "recurso": 2,
    "resultado": 3,
    "aviso": 4,
    "retificacao": 5
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def fixar_seed(seed: int = 42):
    """Garante reprodutibilidade total em Python, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenizar(texto: str) -> list[str]:
    """Tokenizador simples por palavras em minúsculo."""
    if not isinstance(texto, str):
        return []
    palavras = re.findall(r"\b\w+\b", texto.lower())
    return palavras


class Vocabulary:
    """Vocabulário construído exclusivamente com os dados de TREINO."""
    def __init__(self, max_size: int = 2500):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.max_size = max_size
        
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.i2w = {0: self.pad_token, 1: self.unk_token}

    def build_from_texts(self, texts: list[str]):
        counter = Counter()
        for t in texts:
            counter.update(tokenizar(t))
        
        # Pega as palavras mais frequentes até o limite
        mais_comuns = counter.most_common(self.max_size - 2)
        for palavra, _ in mais_comuns:
            if palavra not in self.w2i:
                idx = len(self.w2i)
                self.w2i[palavra] = idx
                self.i2w[idx] = palavra

    def encode(self, texto: str, max_len: int = 128) -> list[int]:
        palavras = tokenizar(texto)
        ids = [self.w2i.get(p, self.w2i[self.unk_token]) for p in palavras[:max_len]]
        # Padding até max_len
        if len(ids) < max_len:
            ids += [self.w2i[self.pad_token]] * (max_len - len(ids))
        return ids

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"w2i": self.w2i}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path):
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.w2i = data["w2i"]
        vocab.i2w = {v: k for k, v in vocab.w2i.items()}
        return vocab


class EditaisDataset(Dataset):
    """Dataset PyTorch para carregar documentos da pós-graduação."""
    def __init__(self, df: pd.DataFrame, vocab: Vocabulary, max_len: int = 128):
        self.textos = df["texto"].tolist()
        self.categorias = df["categoria"].tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        texto = self.textos[idx]
        cat = self.categorias[idx]
        
        token_ids = self.vocab.encode(texto, max_len=self.max_len)
        label_id = LABEL_TO_ID[cat]
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label_id, dtype=torch.long)


def criar_dataloaders(batch_size: int = 8, max_len: int = 128):
    """
    Cria DataLoaders para Treino, Validação e Teste:
      - Treino: shuffle=True (embaralha a cada época)
      - Validação e Teste: shuffle=False (avaliação estável e determinística)
    """
    fixar_seed(SEED)
    
    df_train = pd.read_parquet(CURATED_DIR / "train.parquet")
    df_val = pd.read_parquet(CURATED_DIR / "validation.parquet")
    df_test = pd.read_parquet(CURATED_DIR / "test.parquet")

    # Vocabulário construído ESTRITAMENTE com os textos de treino
    vocab = Vocabulary(max_size=2500)
    vocab.build_from_texts(df_train["texto"].tolist())
    vocab.save(MODELS_DIR / "vocab.json")

    train_ds = EditaisDataset(df_train, vocab, max_len=max_len)
    val_ds = EditaisDataset(df_val, vocab, max_len=max_len)
    test_ds = EditaisDataset(df_test, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


@torch.inference_mode()
def avaliar_modelo(model, loader, loss_fn):
    """
    Função de avaliação isolada:
    - model.eval() desliga dropout;
    - torch.inference_mode() impede o cálculo de gradientes e economiza memória.
    """
    model.eval()
    total_loss = 0.0
    corretos = 0
    total = 0

    for X, y in loader:
        pred = model(X)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * len(X)
        corretos += (pred.argmax(dim=1) == y).sum().item()
        total += len(y)

    loss_media = total_loss / total
    acuracia = corretos / total
    return loss_media, acuracia


def testar_pipeline():
    print("[*] Iniciando teste do Dataset e DataLoader em PyTorch...")
    print("-" * 60)
    
    batch_size = 8
    train_loader, val_loader, test_loader, vocab = criar_dataloaders(batch_size=batch_size)

    print(f"[OK] 1. Vocabulario construído com dados de treino:")
    print(f"        • Tamanho do vocabulário: {len(vocab.w2i)} palavras únicas")
    print(f"        • Arquivo salvo em: models/vocab.json")
    print("-" * 60)

    print(f"[OK] 2. Configuracao dos DataLoaders (Batch size = {batch_size}):")
    print(f"        • Train Loader:      {len(train_loader.dataset)} docs em {len(train_loader)} batches (shuffle=True)")
    print(f"        • Validation Loader: {len(val_loader.dataset)} docs em {len(val_loader)} batches (shuffle=False)")
    print(f"        • Test Loader:       {len(test_loader.dataset)} docs em {len(test_loader)} batches (shuffle=False)")
    print("-" * 60)

    # Inspeciona um batch real extraído do train_loader
    for X_batch, y_batch in train_loader:
        print("[*] 3. Inspecao de 1 Batch Real:")
        print(f"        • Shape dos tensores X: {list(X_batch.shape)} [Batch={X_batch.shape[0]}, Tokens={X_batch.shape[1]}]")
        print(f"        • Shape dos rótulos y:  {list(y_batch.shape)}")
        print(f"        • Exemplos de classes no lote: {[ID_TO_LABEL[i.item()] for i in y_batch[:4]]}")
        break

    print("-" * 60)
    print("🎉 Pipeline de Dados e DataLoader funcionando com determinismo (Seed=42)!")


if __name__ == "__main__":
    testar_pipeline()

"""
Pipeline de Dados em PyTorch: Dataset, Vocabulário e DataLoaders (AcademicIA).
Limpeza inteligente de cabeçalhos institucionais, tokenização n-gram e aumento de dados.
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


def limpar_documento(t: str) -> str:
    """
    Remove ruído institucional repetitivo (UFPI, Ministério, CEP, telefones, números longos)
    para evidenciar o título, tipo e objetivo do documento.
    """
    if not isinstance(t, str):
        return ""
    t = re.sub(r"[\r\n\t]+", " ", t)
    # Remove padrões genéricos de cabeçalho institucional comum a todos os editais
    t = re.sub(r"(?i)minist[eé]rio da educa[cç][aã]o", "", t)
    t = re.sub(r"(?i)universidade federal do piau[ií]", "", t)
    t = re.sub(r"(?i)campus universit[aá]rio petr[oô]nio portella", "", t)
    t = re.sub(r"(?i)bairro ininga", "", t)
    t = re.sub(r"(?i)teresina\s*[-–]\s*pi", "", t)
    t = re.sub(r"(?i)cep:?\s*[\d.-]+", "", t)
    t = re.sub(r"(?i)e-?mail:?\s*[\w@.-]+", "", t)
    t = re.sub(r"(?i)telefone:?\s*[\d() -]+", "", t)
    t = re.sub(r"\b\d{4,}\b", "", t)  # Remove números longos (CPFs, matrículas, inscrições de tabela)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def tokenizar(texto: str) -> list[str]:
    """Tokenizador focado em palavras semânticas com 2 ou mais letras."""
    texto_limpo = limpar_documento(texto)
    palavras = re.findall(r"\b[a-zA-ZáéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ]{2,}\b", texto_limpo.lower())
    return palavras


class Vocabulary:
    """Vocabulário construído exclusivamente com os dados de TREINO."""
    def __init__(self, max_size: int = 2500, min_freq: int = 2):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.i2w = {0: self.pad_token, 1: self.unk_token}

    def build_from_texts(self, texts: list[str]):
        counter = Counter()
        for t in texts:
            counter.update(tokenizar(t))
        
        mais_comuns = counter.most_common(self.max_size - 2)
        for palavra, freq in mais_comuns:
            if freq >= self.min_freq and palavra not in self.w2i:
                idx = len(self.w2i)
                self.w2i[palavra] = idx
                self.i2w[idx] = palavra

    def encode(self, texto: str, max_len: int = 200) -> list[int]:
        palavras = tokenizar(texto)
        ids = [self.w2i.get(p, self.w2i[self.unk_token]) for p in palavras[:max_len]]
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
    """Dataset PyTorch para documentos com suporte a augmentation."""
    def __init__(self, textos: list[str], categorias: list[str], vocab: Vocabulary, max_len: int = 200):
        self.textos = textos
        self.categorias = categorias
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


def criar_dataloaders(batch_size: int = 8, max_len: int = 200):
    """
    Cria DataLoaders para Treino, Validação e Teste com Aumento de Dados Inteligente.
    """
    fixar_seed(SEED)
    
    df_train = pd.read_parquet(CURATED_DIR / "train.parquet")
    df_val = pd.read_parquet(CURATED_DIR / "validation.parquet")
    df_test = pd.read_parquet(CURATED_DIR / "test.parquet")

    # Vocabulário construído ESTRITAMENTE com os textos de treino
    vocab = Vocabulary(max_size=2500, min_freq=2)
    vocab.build_from_texts(df_train["texto"].tolist())
    vocab.save(MODELS_DIR / "vocab.json")

    # Data Augmentation no Treino (janelas deslizantes para robustez a deslocamentos)
    train_textos, train_cats = [], []
    for texto, cat in zip(df_train["texto"], df_train["categoria"]):
        train_textos.append(texto)
        train_cats.append(cat)
        
        tokens = tokenizar(texto)
        # Variação 1: início ligeiramente deslocado (pula introdução)
        if len(tokens) > 40:
            train_textos.append(" ".join(tokens[15:]))
            train_cats.append(cat)
        # Variação 2: início a partir do corpo
        if len(tokens) > 70:
            train_textos.append(" ".join(tokens[30:]))
            train_cats.append(cat)

    train_ds = EditaisDataset(train_textos, train_cats, vocab, max_len=max_len)
    val_ds = EditaisDataset(df_val["texto"].tolist(), df_val["categoria"].tolist(), vocab, max_len=max_len)
    test_ds = EditaisDataset(df_test["texto"].tolist(), df_test["categoria"].tolist(), vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


@torch.inference_mode()
def avaliar_modelo(model, loader, loss_fn):
    """Função de avaliação isolada em modo de inferência."""
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

    print(f"[OK] 1. Vocabulário construído com dados de treino:")
    print(f"        • Tamanho do vocabulário: {len(vocab.w2i)} palavras únicas")
    print(f"        • Arquivo salvo em: models/vocab.json")
    print("-" * 60)

    print(f"[OK] 2. Configuração dos DataLoaders (Batch size = {batch_size}):")
    print(f"        • Train Loader (com Augmentation): {len(train_loader.dataset)} docs em {len(train_loader)} batches")
    print(f"        • Validation Loader:              {len(val_loader.dataset)} docs em {len(val_loader)} batches")
    print(f"        • Test Loader:                    {len(test_loader.dataset)} docs em {len(test_loader)} batches")
    print("-" * 60)

    for X_batch, y_batch in train_loader:
        print("[*] 3. Inspeção de 1 Batch Real:")
        print(f"        • Shape dos tensores X: {list(X_batch.shape)}")
        print(f"        • Shape dos rótulos y:  {list(y_batch.shape)}")
        print(f"        • Exemplos de classes:  {[ID_TO_LABEL[i.item()] for i in y_batch[:4]]}")
        break

    print("-" * 60)
    print("🎉 Pipeline de Dados e DataLoader funcionando com determinismo!")


if __name__ == "__main__":
    testar_pipeline()

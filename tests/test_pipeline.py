"""
Testes automatizados com PyTest para o pipeline do AcademicIA.
Valida integridade dos dados, formato dos tensores (shapes), gradientes e checkpoints.
"""
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import pytest

from src.models.network import TextClassifier
from src.data.dataset import Vocabulary

DATA_DIR = Path("data/curated")
MODELS_DIR = Path("models")


def test_data_integrity_and_no_leakage():
    """Valida integridade das colunas e ausência de vazamento de dados entre os splits."""
    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "validation.parquet"
    test_path = DATA_DIR / "test.parquet"

    assert train_path.exists(), "train.parquet não encontrado"
    assert val_path.exists(), "validation.parquet não encontrado"
    assert test_path.exists(), "test.parquet não encontrado"

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # 1. Esquema e ausência de nulos
    for df in [df_train, df_val, df_test]:
        assert "texto" in df.columns and "categoria" in df.columns
        assert df["texto"].notna().all(), "Há textos com valor nulo"
        assert (df["num_caracteres"] >= 50).all(), "Há textos menores que 50 caracteres"

    # 2. Ausência de sobreposição de textos entre treino e teste (Sem Data Leakage)
    textos_treino = set(df_train["texto"])
    textos_teste = set(df_test["texto"])
    intersecao = textos_treino.intersection(textos_teste)
    assert len(intersecao) == 0, f"Vazamento detectado! {len(intersecao)} textos repetidos entre treino e teste."


def test_model_output_shape():
    """Valida se a rede neural devolve exatamente o tensor [Batch, 6]."""
    vocab_size = 500
    embedding_dim = 32
    hidden_dim = 32
    num_classes = 6

    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )

    batch_size = 5
    seq_len = 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)

    assert logits.shape == (batch_size, num_classes), f"Shape incorreto: esperado {(batch_size, num_classes)}, obteve {logits.shape}"


def test_backward_produces_gradients():
    """Valida se a chamada loss.backward() gera gradientes válidos em todas as camadas."""
    model = TextClassifier(vocab_size=100, embedding_dim=16, hidden_dim=16, num_classes=6)
    x = torch.randint(0, 100, (4, 32))
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss_fn = nn.CrossEntropyLoss()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    # Todos os parâmetros com requires_grad devem ter gradientes numéricos não nulos
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parâmetro {name} não recebeu gradiente"
            assert not torch.isnan(param.grad).any(), f"Gradiente NaN detectado em {name}"


def test_checkpoint_reproducibility():
    """Valida se um checkpoint salvo e recarregado produz inferência 100% determinística."""
    model1 = TextClassifier(vocab_size=50, embedding_dim=16, hidden_dim=16, num_classes=6)
    model1.eval()

    temp_path = MODELS_DIR / "temp_test_ckpt.pt"
    torch.save(model1.state_dict(), temp_path)

    model2 = TextClassifier(vocab_size=50, embedding_dim=16, hidden_dim=16, num_classes=6)
    model2.load_state_dict(torch.load(temp_path, weights_only=True))
    model2.eval()

    x = torch.randint(0, 50, (2, 20))
    with torch.inference_mode():
        out1 = model1(x)
        out2 = model2(x)

    # Remove o arquivo temporário
    if temp_path.exists():
        temp_path.unlink()

    assert torch.allclose(out1, out2, atol=1e-6), "O modelo recarregado divergiu na inferência!"


def test_scraped_data_hierarchy_and_sources():
    """Valida a persistência estruturada dos cursos do Piauí e a hierarquia do scraper."""
    import json

    # 1. Valida banco de fontes oficial das 4 universidades
    cursos_path = Path("data/cursos_piaui.json")
    assert cursos_path.exists(), "data/cursos_piaui.json não foi encontrado"

    with open(cursos_path, "r", encoding="utf-8") as f:
        cursos = json.load(f)

    assert set(cursos.keys()) == {"UESPI", "UFPI", "IFPI", "UFDPAR"}
    total_cursos = sum(len(v) for v in cursos.values())
    assert total_cursos == 83, f"Esperado 83 cursos, encontrado {total_cursos}"

    # 2. Valida catálogo de scraping e hierarquia Faculdade > Curso
    catalogo_path = Path("data/scraped/catalogo_documentos.json")
    if catalogo_path.exists():
        with open(catalogo_path, "r", encoding="utf-8") as f:
            catalogo = json.load(f)

        assert len(catalogo) > 0, "Catálogo de documentos coletados está vazio"
        for doc in catalogo:
            assert doc["faculdade"] in {"UESPI", "UFPI", "IFPI", "UFDPAR"}
            assert "pasta_curso" in doc and doc["pasta_curso"]
            assert "caminho_relativo" in doc
            assert "classificacao_ia" in doc
            # Garante que o caminho segue estritamente data/scraped/{faculdade}/{curso}/
            partes = Path(doc["caminho_relativo"]).parts
            assert partes[0] == "data" and partes[1] == "scraped"
            assert partes[2] == doc["faculdade"]
            assert partes[3] == doc["pasta_curso"]


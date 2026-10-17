"""
Módulo de Inferência em Produção (AcademicIA).
Carrega o checkpoint PyTorch treinado (models/best.pt) e o vocabulário (models/vocab.json).
Executa previsões determinísticas com @torch.inference_mode() e sem cálculo de gradientes.
"""
import sys
import io
import re
import json
import time
import argparse
from pathlib import Path
from typing import Union, BinaryIO

import torch
from pypdf import PdfReader

# Encoding seguro para Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "best.pt"
DEFAULT_VOCAB = MODELS_DIR / "vocab.json"

# Mapeamento oficial das 6 classes do AcademicIA
ID_TO_LABEL = {
    0: "edital",
    1: "homologacao",
    2: "recurso",
    3: "resultado",
    4: "aviso",
    5: "retificacao"
}
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}


def limpar_texto(texto: str) -> str:
    """Normaliza quebras de linha e múltiplos espaços."""
    if not texto:
        return ""
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return texto.strip()


class Predictor:
    """
    Engine de Inferência Otimizada em CPU.
    Carrega o modelo uma única vez na memória para responder a requisições com baixa latência.
    """
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        vocab_path: Path = DEFAULT_VOCAB,
        max_len: int = 128,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.max_len = max_len
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado em: {self.checkpoint_path}")
        if not self.vocab_path.exists():
            raise FileNotFoundError(f"Vocabulário não encontrado em: {self.vocab_path}")

        # 1. Carrega o vocabulário oficial de treino
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        self.w2i = vocab_data["w2i"]
        self.pad_id = self.w2i.get("<pad>", 0)
        self.unk_id = self.w2i.get("<unk>", 1)
        self.vocab_size = len(self.w2i)

        # 2. Carrega o checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # 3. Importa e instancia a arquitetura da rede neural
        sys.path.insert(0, str(BASE_DIR))
        from src.models.network import TextClassifier

        emb_dim = checkpoint.get("embedding_dim", 128)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        num_classes = checkpoint.get("num_classes", 6)

        self.model = TextClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_prob=0.0
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.metadata = {
            "checkpoint": self.checkpoint_path.name,
            "epoch": checkpoint.get("epoch", None),
            "val_loss": checkpoint.get("val_loss", None),
            "val_f1": checkpoint.get("val_f1", None),
            "vocab_size": self.vocab_size,
            "embedding_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "device": str(self.device)
        }

    def encode(self, texto: str) -> list[int]:
        """Tokeniza e converte texto em lista de IDs com padding fixo."""
        palavras = re.findall(r"\b\w+\b", texto.lower())
        ids = [self.w2i.get(p, self.unk_id) for p in palavras[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [self.pad_id] * (self.max_len - len(ids))
        return ids

    @torch.inference_mode()
    def predict_text(self, texto: str) -> dict:
        """
        Executa inferência direta sobre uma string de texto.
        Garante modo de inferência (sem autograd / gradientes).
        """
        inicio = time.perf_counter()
        texto_limpo = limpar_texto(texto)

        if len(texto_limpo) < 10:
            return {
                "erro": "Texto muito curto (mínimo de 10 caracteres)",
                "classe_predita": "indeterminado",
                "confianca": 0.0,
                "probabilidades": {}
            }

        token_ids = self.encode(texto_limpo)
        tensor_in = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        logits = self.model(tensor_in)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = ID_TO_LABEL[pred_idx]
        confianca = probs[pred_idx]

        latencia_ms = round((time.perf_counter() - inicio) * 1000, 2)
        distribuicao = {ID_TO_LABEL[i]: round(probs[i] * 100, 2) for i in range(len(probs))}

        return {
            "classe_predita": pred_label,
            "confianca_percentual": round(confianca * 100, 2),
            "probabilidade_bruta": round(confianca, 4),
            "distribuicao_probabilidades": distribuicao,
            "latencia_ms": latencia_ms,
            "tokens_processados": min(len(re.findall(r"\b\w+\b", texto_limpo.lower())), self.max_len),
            "trecho_analisado": texto_limpo[:300] + ("..." if len(texto_limpo) > 300 else "")
        }

    def predict_pdf(self, pdf_source: Union[Path, str, BinaryIO]) -> dict:
        """
        Extrai o texto da primeira página do PDF e executa a inferência.
        Suporta caminho de arquivo ou fluxo de bytes (UploadFile).
        """
        inicio = time.perf_counter()
        try:
            reader = PdfReader(pdf_source)
            num_paginas = len(reader.pages)
            if num_paginas == 0:
                return {"erro": "PDF vazio", "classe_predita": "indeterminado", "confianca": 0.0}

            texto_acumulado = reader.pages[0].extract_text() or ""
            if len(limpar_texto(texto_acumulado)) < 80 and num_paginas > 1:
                texto_acumulado += " " + (reader.pages[1].extract_text() or "")

            texto_limpo = limpar_texto(texto_acumulado)
            if not texto_limpo:
                return {
                    "erro": "PDF escaneado ou sem camada de texto extraível",
                    "classe_predita": "indeterminado",
                    "confianca": 0.0,
                    "paginas": num_paginas
                }

            resultado = self.predict_text(texto_limpo)
            resultado["paginas_totais"] = num_paginas
            resultado["latencia_total_ms"] = round((time.perf_counter() - inicio) * 1000, 2)
            return resultado

        except Exception as e:
            return {
                "erro": f"Falha ao processar arquivo PDF: {str(e)}",
                "classe_predita": "indeterminado",
                "confianca": 0.0
            }


# Instância singleton para uso em APIs e scripts
_predictor_instance = None

def get_predictor() -> Predictor:
    """Retorna uma instância única (Singleton) do Predictor na memória."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance


def main():
    parser = argparse.ArgumentParser(description="Inferência de Editais com PyTorch - AcademicIA")
    parser.add_argument("--texto", type=str, help="Texto do edital/documento para classificar")
    parser.add_argument("--pdf", type=str, help="Caminho para arquivo PDF")
    args = parser.parse_args()

    predictor = get_predictor()
    print("=" * 60)
    print("🤖 MOTOR DE INFERÊNCIA - AcademicIA (Deep Learning)")
    print(f"   Checkpoint: {predictor.metadata['checkpoint']}")
    print(f"   Vocabulário: {predictor.metadata['vocab_size']} termos")
    print(f"   Dispositivo: {predictor.metadata['device']}")
    print("=" * 60)

    if args.pdf:
        pdf_path = Path(args.pdf)
        print(f"Analisando arquivo PDF: {pdf_path.name}")
        res = predictor.predict_pdf(pdf_path)
    elif args.texto:
        res = predictor.predict_text(args.texto)
    else:
        # Exemplo padrão com um trecho de resultado
        exemplo = "A Pró-Reitoria de Pós-Graduação da Universidade Federal do Piauí torna público o resultado final da seleção do mestrado após julgamento dos recursos."
        print(f"Exemplo de entrada: '{exemplo}'\n")
        res = predictor.predict_text(exemplo)

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

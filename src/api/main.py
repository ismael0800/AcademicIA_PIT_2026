"""
API REST em FastAPI e Interface de Demonstração (AcademicIA).
Provê contratos estritos de entrada/saída (Pydantic), health check,
upload de arquivos PDF e interface web interativa em tempo real.
"""
import sys
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Adiciona raiz do projeto ao path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.models.inference import get_predictor, ID_TO_LABEL

app = FastAPI(
    title="AcademicIA - Classificador de Documentos da Pós-Graduação",
    description="API de Inferência em Deep Learning para triagem e classificação de editais, resultados, recursos e homologações (Stricto Sensu).",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Habilita CORS para testes locais e integrações externas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Contratos de Dados (Pydantic Schemas)
# ==============================================================================

class PredictionInput(BaseModel):
    texto: str = Field(
        ...,
        min_length=10,
        max_length=20000,
        description="Texto extraído ou copiado do documento a ser classificado",
        examples=["EDITAL Nº 01/2026 - O Diretor do Centro de Ciências da Universidade torna público o processo seletivo para o Mestrado."]
    )

class PredictionOutput(BaseModel):
    status: str = Field("sucesso", description="Status da inferência")
    classe_predita: str = Field(..., description="Categoria predita entre as 6 classes oficiais")
    confianca_percentual: float = Field(..., description="Nível de confiança da previsão (%)")
    probabilidade_bruta: float = Field(..., description="Probabilidade normalizada (0.0 a 1.0)")
    distribuicao_probabilidades: Dict[str, float] = Field(..., description="Probabilidade percentual de cada uma das 6 classes")
    latencia_ms: float = Field(..., description="Tempo de inferência em milissegundos")
    tokens_processados: int = Field(..., description="Quantidade de tokens analisados pela rede")
    trecho_analisado: str = Field(..., description="Amostra do texto processado")

class HealthResponse(BaseModel):
    status: str
    servico: str
    versao: str
    modelo: str
    dispositivo: str
    tamanho_vocabulario: int
    classes_suportadas: list[str]


# ==============================================================================
# Rotas da API REST
# ==============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
def health_check():
    """Verifica a integridade da API e se o checkpoint PyTorch está pronto para inferência."""
    try:
        predictor = get_predictor()
        return HealthResponse(
            status="ok",
            servico="AcademicIA Inference API",
            versao="1.1.0",
            modelo=predictor.metadata["checkpoint"],
            dispositivo=predictor.metadata["device"],
            tamanho_vocabulario=predictor.metadata["vocab_size"],
            classes_suportadas=list(ID_TO_LABEL.values())
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modelo indisponível ou falha de inicialização: {str(e)}"
        )


@app.post("/predict", response_model=PredictionOutput, tags=["Inferência"])
def prever_texto(payload: PredictionInput):
    """Classifica o texto de um documento acadêmico em tempo real."""
    predictor = get_predictor()
    resultado = predictor.predict_text(payload.texto)
    
    if "erro" in resultado and resultado["classe_predita"] == "indeterminado":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=resultado["erro"])
        
    return PredictionOutput(
        status="sucesso",
        classe_predita=resultado["classe_predita"],
        confianca_percentual=resultado["confianca_percentual"],
        probabilidade_bruta=resultado["probabilidade_bruta"],
        distribuicao_probabilidades=resultado["distribuicao_probabilidades"],
        latencia_ms=resultado["latencia_ms"],
        tokens_processados=resultado["tokens_processados"],
        trecho_analisado=resultado["trecho_analisado"]
    )


@app.post("/predict/pdf", tags=["Inferência"])
async def prever_pdf(arquivo: UploadFile = File(..., description="Arquivo PDF do edital ou processo seletivo")):
    """Recebe um arquivo PDF por upload, extrai o texto da primeira página e executa a inferência."""
    if not arquivo.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arquivo inválido. Apenas arquivos PDF (.pdf) são aceitos."
        )

    conteudo = await arquivo.read()
    if len(conteudo) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo enviado está vazio (0 bytes)."
        )

    # Valida assinatura mágica de PDF
    if not conteudo.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo enviado não possui cabeçalho PDF válido."
        )

    predictor = get_predictor()
    buffer = io.BytesIO(conteudo)
    buffer.name = arquivo.filename  # Permite enriquecimento contextual de metadados a partir do nome do arquivo
    resultado = predictor.predict_pdf(buffer)

    if "erro" in resultado and resultado.get("classe_predita") == "indeterminado":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "indeterminado",
                "arquivo": arquivo.filename,
                "detalhe": resultado.get("erro"),
                "paginas": resultado.get("paginas", 0)
            }
        )

    resultado["status"] = "sucesso"
    resultado["arquivo"] = arquivo.filename
    resultado["tamanho_bytes"] = len(conteudo)
    return resultado


# ==============================================================================
# Interface Web Profissional (HTML5 + CSS Moderno + Vanilla JS)
# ==============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
def interface_web():
    """Serve a interface gráfica com design profissional e clean para demonstração e banca examinadora."""
    html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AcademicIA — Triagem Inteligente de Documentos Acadêmicos</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #111827;
      --card-subtle: #1f293d;
      --accent: #2563eb;
      --accent-light: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.12);
      --text: #f9fafb;
      --text-muted: #9ca3af;
      --text-dim: #6b7280;
      --border: #1f2937;
      --border-hover: #374151;
      --border-accent: rgba(56, 189, 248, 0.3);
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #8b5cf6;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      background-image: radial-gradient(circle at 50% -10%, rgba(37, 99, 235, 0.12), transparent 50%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      line-height: 1.5;
    }

    /* Top Navigation Bar */
    .navbar {
      border-bottom: 1px solid var(--border);
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 50;
    }
    .navbar-inner {
      max-width: 1040px;
      margin: 0 auto;
      padding: 0.9rem 1.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: var(--text);
    }
    .brand-logo {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: linear-gradient(135deg, #38bdf8, #2563eb);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
    }
    .brand-title {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-badge {
      font-size: 0.7rem;
      font-weight: 600;
      padding: 0.15rem 0.5rem;
      border-radius: 9999px;
      background: rgba(56, 189, 248, 0.1);
      color: var(--accent-light);
      border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .nav-actions {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .status-indicator {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(16, 185, 129, 0.2);
      padding: 0.3rem 0.75rem;
      border-radius: 9999px;
    }
    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    .nav-link {
      color: var(--text-muted);
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 500;
      transition: color 0.2s;
    }
    .nav-link:hover { color: var(--text); }

    /* Main Container */
    .main-wrap {
      flex: 1;
      max-width: 920px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 3rem 1.25rem;
      width: 100%;
    }

    /* Hero Section */
    .hero {
      text-align: center;
      margin-bottom: 2rem;
    }
    .hero h1 {
      font-size: 2.1rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      color: var(--text);
      margin-bottom: 0.5rem;
    }
    .hero p {
      color: var(--text-muted);
      font-size: 0.98rem;
      max-width: 600px;
      margin: 0 auto;
    }

    /* Workspace Card */
    .workspace-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1.5rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
      margin-bottom: 1.5rem;
    }

    /* Tabs */
    .tabs-header {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.25rem;
      padding-bottom: 0.25rem;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 0.65rem 1.15rem;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      border-radius: 8px;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    .tab-btn:hover {
      color: var(--text);
      background: rgba(255, 255, 255, 0.03);
    }
    .tab-btn.active {
      color: var(--text);
      background: var(--card-subtle);
      border: 1px solid var(--border-hover);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* Dropzone PDF */
    .dropzone {
      border: 2px dashed var(--border-hover);
      border-radius: 12px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      background: rgba(17, 24, 39, 0.4);
      transition: all 0.2s ease;
      margin-bottom: 1rem;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent-light);
      background: rgba(56, 189, 248, 0.03);
    }
    .dropzone-icon {
      font-size: 2.2rem;
      margin-bottom: 0.5rem;
      display: inline-block;
    }
    .dropzone-title {
      font-size: 0.98rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.25rem;
    }
    .dropzone-desc {
      font-size: 0.82rem;
      color: var(--text-dim);
    }

    /* File Selected Box */
    .file-box {
      display: none;
      background: var(--card-subtle);
      border: 1px solid var(--border-hover);
      border-radius: 10px;
      padding: 0.9rem 1.15rem;
      margin-bottom: 1rem;
      align-items: center;
      justify-content: space-between;
    }
    .file-info {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .file-name {
      font-weight: 600;
      font-size: 0.88rem;
      color: var(--text);
    }
    .file-meta {
      font-size: 0.75rem;
      color: var(--text-dim);
    }
    .btn-close {
      background: none;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      padding: 0.3rem 0.5rem;
      font-size: 1rem;
      border-radius: 6px;
      transition: all 0.2s;
    }
    .btn-close:hover {
      color: var(--danger);
      background: rgba(239, 68, 68, 0.1);
    }

    /* Textarea */
    textarea {
      width: 100%;
      height: 150px;
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.9rem 1rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.88rem;
      resize: vertical;
      line-height: 1.6;
      transition: border-color 0.2s;
      margin-bottom: 0.4rem;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent-light);
    }
    .textarea-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.75rem;
      color: var(--text-dim);
      margin-bottom: 1.25rem;
    }

    /* Samples Strip */
    .sample-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }
    .sample-label {
      font-size: 0.78rem;
      color: var(--text-dim);
      font-weight: 500;
    }
    .sample-btn {
      background: transparent;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.3rem 0.65rem;
      border-radius: 6px;
      font-size: 0.76rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .sample-btn:hover {
      border-color: var(--accent-light);
      color: var(--text);
      background: rgba(56, 189, 248, 0.05);
    }

    /* Action Buttons */
    .btn-primary {
      background: #2563eb;
      color: #ffffff;
      border: none;
      padding: 0.75rem 1.6rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-primary:hover {
      background: #1d4ed8;
      transform: translateY(-1px);
    }
    .btn-primary:active { transform: translateY(0); }
    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    .btn-secondary {
      background: transparent;
      color: var(--text-muted);
      border: 1px solid var(--border);
      padding: 0.75rem 1.2rem;
      border-radius: 8px;
      font-weight: 500;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      background: var(--card-subtle);
      color: var(--text);
      border-color: var(--border-hover);
    }

    /* Result Card */
    #resultado-card {
      display: none;
      background: var(--card-bg);
      border: 1px solid var(--border-accent);
      border-radius: 14px;
      padding: 1.5rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      margin-bottom: 1.5rem;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .res-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.25rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }
    .res-title-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
    }
    .res-kicker {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-dim);
      font-weight: 600;
    }
    .res-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 1.25rem;
      font-weight: 700;
      padding: 0.35rem 0.85rem;
      border-radius: 8px;
      width: fit-content;
    }
    .badge-edital { background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-homologacao { background: rgba(139, 92, 246, 0.12); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-resultado { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-recurso { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-aviso { background: rgba(99, 102, 241, 0.12); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-retificacao { background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); }
    .badge-indeterminado { background: rgba(156, 163, 175, 0.12); color: #9ca3af; border: 1px solid rgba(156, 163, 175, 0.3); }

    .confidence-panel {
      text-align: right;
    }
    .confidence-score {
      font-size: 1.8rem;
      font-weight: 800;
      color: #38bdf8;
      line-height: 1;
    }
    .confidence-text {
      font-size: 0.72rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 0.2rem;
    }

    /* Meta Strip */
    .res-meta-strip {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-bottom: 1.25rem;
    }
    .res-meta-strip strong { color: var(--text); }

    /* Distributions */
    .dist-title {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 0.75rem;
      display: flex;
      justify-content: space-between;
    }
    .dist-list {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      margin-bottom: 1.25rem;
    }
    .dist-item {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.5rem 0.75rem;
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }
    .dist-item.winner {
      border-color: rgba(56, 189, 248, 0.3);
      background: rgba(56, 189, 248, 0.03);
    }
    .dist-row-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
    }
    .track {
      background: #1f293d;
      border-radius: 9999px;
      height: 6px;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.5s ease;
    }

    /* Snippet Box */
    .snippet-box {
      background: #090d16;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      color: #94a3b8;
      line-height: 1.5;
      max-height: 110px;
      overflow-y: auto;
      margin-bottom: 1.25rem;
    }

    /* Actions */
    .res-actions {
      display: flex;
      gap: 0.5rem;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    /* Alert */
    .alert-banner {
      display: none;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 0.75rem 1rem;
      border-radius: 8px;
      margin-bottom: 1.25rem;
      font-size: 0.85rem;
      justify-content: space-between;
      align-items: center;
    }

    /* Toast */
    .toast {
      position: fixed;
      bottom: 1.5rem;
      right: 1.5rem;
      background: #1f293d;
      border: 1px solid var(--accent-light);
      color: #ffffff;
      padding: 0.65rem 1.15rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 500;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.25s ease;
      pointer-events: none;
      z-index: 100;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    /* Spinner */
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 50%;
      border-top-color: #ffffff;
      animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Model Specs Accordion (Tucked away discreetly) */
    .specs-accordion {
      margin-top: 2rem;
      border-top: 1px solid var(--border);
      padding-top: 1rem;
    }
    .specs-accordion summary {
      cursor: pointer;
      font-size: 0.78rem;
      color: var(--text-dim);
      outline: none;
      user-select: none;
      transition: color 0.2s;
    }
    .specs-accordion summary:hover {
      color: var(--text-muted);
    }
    .specs-content {
      margin-top: 0.75rem;
      background: rgba(17, 24, 39, 0.5);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      font-size: 0.78rem;
      color: var(--text-muted);
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
    }
    .spec-item span {
      display: block;
      color: var(--text-dim);
      font-size: 0.7rem;
      text-transform: uppercase;
    }
    .spec-item strong {
      color: var(--text);
      font-size: 0.85rem;
    }

    /* Footer */
    footer {
      border-top: 1px solid var(--border);
      padding: 1.25rem 1.5rem;
      text-align: center;
      font-size: 0.8rem;
      color: var(--text-dim);
      background: var(--bg);
    }
    footer a { color: var(--text-muted); text-decoration: none; }
    footer a:hover { color: var(--accent-light); }
  </style>
</head>
<body>

  <!-- Navigation -->
  <nav class="navbar">
    <div class="navbar-inner">
      <a href="/" class="brand">
        <div class="brand-logo">🎓</div>
        <span class="brand-title">AcademicIA</span>
        <span class="brand-badge">Produção</span>
      </a>
      <div class="nav-actions">
        <div class="status-indicator">
          <span class="status-dot"></span>
          <span id="nav-status-text">Online</span>
        </div>
        <a href="/docs" target="_blank" class="nav-link">Documentação API ↗</a>
      </div>
    </div>
  </nav>

  <!-- Content -->
  <main class="main-wrap">
    <header class="hero">
      <h1>Triagem Inteligente de Documentos</h1>
      <p>Classificação automatizada de editais, homologações, resultados, recursos e retificações da pós-graduação Stricto Sensu.</p>
    </header>

    <!-- Error Banner -->
    <div id="error-alert" class="alert-banner">
      <span id="error-msg">Ocorreu um erro ao processar.</span>
      <button onclick="fecharAlerta()" style="background:none;border:none;color:#fca5a5;cursor:pointer;">✕</button>
    </div>

    <!-- Workspace Input Card -->
    <div class="workspace-card">
      <div class="tabs-header">
        <button class="tab-btn active" id="btn-tab-pdf" onclick="switchTab('tab-pdf')">
          📄 Upload de Arquivo PDF
        </button>
        <button class="tab-btn" id="btn-tab-texto" onclick="switchTab('tab-texto')">
          📝 Digitar ou Colar Texto
        </button>
      </div>

      <!-- Aba 1: PDF (Padrão) -->
      <div id="tab-pdf" class="tab-content active">
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept="application/pdf" style="display:none" onchange="tratarArquivo(this)">
          <div class="dropzone-icon">📥</div>
          <div class="dropzone-title">Arraste e solte o arquivo PDF aqui</div>
          <div class="dropzone-desc">Ou clique para procurar em seu computador (análise instantânea em &lt; 10ms)</div>
        </div>

        <div class="file-box" id="file-box">
          <div class="file-info">
            <span style="font-size: 1.4rem;">📄</span>
            <div>
              <div class="file-name" id="file-name-label">documento.pdf</div>
              <div class="file-meta" id="file-size-label">0 KB</div>
            </div>
          </div>
          <button class="btn-close" onclick="removerArquivo()" title="Remover arquivo">✕</button>
        </div>

        <button class="btn-primary" id="btn-classificar-pdf" onclick="classificarPDF()">
          <span>⚡ Analisar Documento PDF</span>
        </button>
      </div>

      <!-- Aba 2: Texto -->
      <div id="tab-texto" class="tab-content">
        <textarea id="input-texto" placeholder="Cole o texto da primeira página do edital, homologação, recurso ou resultado..." oninput="atualizarContador()"></textarea>
        <div class="textarea-footer">
          <span id="char-count">0 caracteres</span>
          <button class="btn-secondary" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="limparTexto()">Limpar</button>
        </div>

        <div style="display: flex; gap: 0.75rem; align-items: center;">
          <button class="btn-primary" id="btn-classificar-texto" onclick="classificarTexto()">
            <span>⚡ Classificar Texto</span>
          </button>
        </div>

        <!-- Exemplos Rápidos -->
        <div class="sample-row">
          <span class="sample-label">Amostras de teste:</span>
          <button class="sample-btn" onclick="carregarExemplo('edital')">Edital 2026</button>
          <button class="sample-btn" onclick="carregarExemplo('homologacao')">Homologação</button>
          <button class="sample-btn" onclick="carregarExemplo('resultado')">Resultado Final</button>
          <button class="sample-btn" onclick="carregarExemplo('recurso')">Recurso</button>
          <button class="sample-btn" onclick="carregarExemplo('aviso')">Aviso</button>
          <button class="sample-btn" onclick="carregarExemplo('retificacao')">Retificação</button>
        </div>
      </div>
    </div>

    <!-- Card de Resultado -->
    <div id="resultado-card">
      <div class="res-top">
        <div class="res-title-group">
          <span class="res-kicker">Classificação Identificada:</span>
          <div id="res-badge" class="res-badge badge-edital">
            <span id="res-icon">📜</span>
            <span id="res-label">EDITAL</span>
          </div>
        </div>
        <div class="confidence-panel">
          <div class="confidence-score" id="res-conf">98.5%</div>
          <div class="confidence-text">Grau de Certeza</div>
        </div>
      </div>

      <!-- Resumo Operacional -->
      <div class="res-meta-strip">
        <div>Origem: <strong id="res-origem">Arquivo PDF</strong></div>
        <div>Processamento: <strong id="res-lat">8.2 ms</strong></div>
        <div>Tokens Lidos: <strong id="res-tokens">142</strong></div>
      </div>

      <!-- Distribuição de Probabilidades -->
      <div class="dist-title">
        <span>Distribuição de Probabilidades</span>
        <span style="font-size: 0.72rem; color: var(--text-dim);">6 Categorias Oficiais</span>
      </div>
      <div class="dist-list" id="dist-container"></div>

      <!-- Trecho Inspecionado -->
      <div class="dist-title" style="margin-bottom: 0.4rem;">Trecho Textual Analisado</div>
      <div class="snippet-box" id="res-trecho">Aguardando dados...</div>

      <!-- Ações -->
      <div class="res-actions">
        <button class="btn-secondary" onclick="copiarResumo()">📋 Copiar Resumo</button>
        <button class="btn-secondary" onclick="baixarJSON()">📥 Baixar JSON</button>
        <button class="btn-secondary" onclick="fecharResultado()">✕ Fechar</button>
      </div>
    </div>

    <!-- Especificações do Modelo (Discreto, sob demanda) -->
    <div class="specs-accordion">
      <details>
        <summary>ℹ️ Detalhes Técnicos e Métricas do Modelo</summary>
        <div class="specs-content">
          <div class="spec-item">
            <span>Arquitetura</span>
            <strong>TextCNN + Self-Attention</strong>
          </div>
          <div class="spec-item">
            <span>Acurácia de Teste</span>
            <strong>78.26%</strong>
          </div>
          <div class="spec-item">
            <span>F1-Score Ponderado</span>
            <strong>0.7899</strong>
          </div>
          <div class="spec-item">
            <span>Motor de Execução</span>
            <strong>PyTorch CPU (best.pt)</strong>
          </div>
          <div class="spec-item">
            <span>Tamanho do Vocabulário</span>
            <strong id="spec-vocab">1.673 termos</strong>
          </div>
          <div class="spec-item">
            <span>Instituições Cobertas</span>
            <strong>UFPI, UESPI, IFPI e UFDPAR</strong>
          </div>
        </div>
      </details>
    </div>
  </main>

  <div id="toast" class="toast">Informações copiadas!</div>

  <footer>
    AcademicIA • Triagem de Documentos da Pós-Graduação • <a href="/docs" target="_blank">Swagger API</a> • <a href="/redoc" target="_blank">ReDoc</a>
  </footer>

  <script>
    const EXEMPLOS = {
      edital: "UNIVERSIDADE FEDERAL DO PIAUÍ - UFPI\\nPRÓ-REITORIA DE ENSINO DE PÓS-GRADUAÇÃO\\nEDITAL Nº 01/2026 - SELEÇÃO PÚBLICA PARA O CURSO DE MESTRADO EM COMUNICAÇÃO\\nO Diretor do Centro no uso de suas atribuições legais torna pública a abertura de inscrições para o processo seletivo do Programa de Pós-Graduação, com oferta de vagas regulares.",
      homologacao: "PROGRAMA DE PÓS-GRADUAÇÃO EM ESTUDOS LITERÁRIOS - PPGEL/UFPI\\nHOMOLOGAÇÃO DAS INSCRIÇÕES - PROCESSO SELETIVO TURMA 2026\\nA Comissão de Seleção torna pública a relação oficial de candidatos com inscrições homologadas (deferidas e indeferidas) após a análise tempestiva da documentação exigida.",
      resultado: "COORDENAÇÃO DO PROGRAMA DE PÓS-GRADUAÇÃO EM CIÊNCIA DOS MATERIAIS\\nDIVULGAÇÃO DO RESULTADO FINAL DA PROVA ESCRITA E ANÁLISE DE CURRÍCULO\\nA Comissão de Seleção torna pública a lista final de candidatos aprovados e classificados por ordem decrescente de pontuação no processo seletivo 2026.",
      recurso: "ILUSTRÍSSIMO SENHOR PRESIDENTE DA COMISSÃO DE SELEÇÃO\\nFORMULÁRIO DE INTERPOSIÇÃO DE RECURSO ADMINISTRATIVO CONTRA O RESULTADO PRELIMINAR\\nO candidato vem respeitosamente à presença de Vossa Senhoria interpor RECURSO contra a pontuação da prova escrita com base nas justificativas fundamentadas anexas.",
      aviso: "AVISO DE CONVOCAÇÃO PARA MATRÍCULA INSTITUCIONAL - MESTRADO E DOUTORADO\\nA Coordenação do Programa de Pós-Graduação convoca todos os candidatos aprovados em 1ª chamada no processo seletivo a comparecerem para entrega de documentos e efetivação de matrícula.",
      retificacao: "RETIFICAÇÃO Nº 01 AO EDITAL Nº 03/2026 - PROCESSO SELETIVO PPGEDUCAÇÃO\\nA Direção torna pública a alteração do cronograma de seleção. Onde se lê: 'Data limite: 15 de março', leia-se: 'Data limite: 22 de março'. Ficam inalteradas as demais disposições."
    };

    const CORES = {
      edital: "#38bdf8",
      homologacao: "#a78bfa",
      resultado: "#34d399",
      recurso: "#fbbf24",
      aviso: "#818cf8",
      retificacao: "#fb7185",
      indeterminado: "#9ca3af"
    };

    const ICONES = {
      edital: "📜",
      homologacao: "📋",
      resultado: "📊",
      recurso: "⚖️",
      aviso: "📢",
      retificacao: "✏️",
      indeterminado: "📄"
    };

    let ultimoResultado = null;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      if (tabId === 'tab-pdf') {
        document.getElementById('btn-tab-pdf').classList.add('active');
      } else {
        document.getElementById('btn-tab-texto').classList.add('active');
      }
      document.getElementById(tabId).classList.add('active');
    }

    function atualizarContador() {
      const len = document.getElementById('input-texto').value.length;
      document.getElementById('char-count').textContent = len + ' caracteres';
    }

    function limparTexto() {
      document.getElementById('input-texto').value = '';
      atualizarContador();
    }

    function carregarExemplo(tipo) {
      if (EXEMPLOS[tipo]) {
        document.getElementById('input-texto').value = EXEMPLOS[tipo];
        atualizarContador();
        switchTab('tab-texto');
      }
    }

    // Drag and drop
    const dropzone = document.getElementById('dropzone');
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.toLowerCase().endsWith('.pdf')) {
        document.getElementById('file-input').files = files;
        tratarArquivo(document.getElementById('file-input'));
      } else {
        mostrarErro('Por favor, selecione um arquivo em formato PDF.');
      }
    });

    function tratarArquivo(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        document.getElementById('file-name-label').textContent = file.name;
        const kb = (file.size / 1024).toFixed(1);
        const mb = (file.size / (1024 * 1024)).toFixed(2);
        document.getElementById('file-size-label').textContent = file.size > 1048576 ? `${mb} MB` : `${kb} KB`;
        
        document.getElementById('dropzone').style.display = 'none';
        document.getElementById('file-box').style.display = 'flex';
      }
    }

    function removerArquivo() {
      document.getElementById('file-input').value = '';
      document.getElementById('dropzone').style.display = 'block';
      document.getElementById('file-box').style.display = 'none';
    }

    function mostrarErro(msg) {
      document.getElementById('error-msg').textContent = msg;
      document.getElementById('error-alert').style.display = 'flex';
    }

    function fecharAlerta() {
      document.getElementById('error-alert').style.display = 'none';
    }

    function mostrarToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2500);
    }

    function fecharResultado() {
      document.getElementById('resultado-card').style.display = 'none';
    }

    function renderizarResultado(data, tipo) {
      ultimoResultado = data;
      fecharAlerta();

      const card = document.getElementById('resultado-card');
      const badge = document.getElementById('res-badge');
      const classe = data.classe_predita || 'indeterminado';

      badge.className = 'res-badge badge-' + classe;
      document.getElementById('res-icon').textContent = ICONES[classe] || '📄';
      document.getElementById('res-label').textContent = classe.toUpperCase();

      const conf = (data.confianca_percentual || 0).toFixed(1) + '%';
      document.getElementById('res-conf').textContent = conf;
      document.getElementById('res-conf').style.color = CORES[classe] || '#38bdf8';

      const lat = (data.latencia_ms || data.latencia_total_ms || 0).toFixed(1) + ' ms';
      document.getElementById('res-lat').textContent = lat;

      document.getElementById('res-tokens').textContent = (data.tokens_processados || 0) + ' tokens';
      document.getElementById('res-origem').textContent = data.arquivo ? data.arquivo : 'Texto Manual';

      document.getElementById('res-trecho').textContent = data.trecho_analisado || data.trecho_inicial || 'Sem trecho.';

      // Distribuição ordenada
      const dist = data.distribuicao_probabilidades || {};
      const sorted = Object.entries(dist).sort((a, b) => b[1] - a[1]);
      const container = document.getElementById('dist-container');
      container.innerHTML = '';

      sorted.forEach(([cls, pct]) => {
        const isWinner = (cls === classe);
        const cor = CORES[cls] || '#38bdf8';
        const icone = ICONES[cls] || '•';

        container.innerHTML += `
          <div class="dist-item ${isWinner ? 'winner' : ''}">
            <div class="dist-row-labels">
              <span style="font-weight: 600; color: ${isWinner ? cor : 'var(--text)'};">
                ${icone} ${cls.toUpperCase()} ${isWinner ? '★' : ''}
              </span>
              <strong style="color: ${isWinner ? cor : 'var(--text-muted)'};">${pct}%</strong>
            </div>
            <div class="track">
              <div class="fill" style="width: ${pct}%; background: ${cor};"></div>
            </div>
          </div>
        `;
      });

      card.style.display = 'block';
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async function classificarPDF() {
      const fileInput = document.getElementById('file-input');
      if (!fileInput.files || !fileInput.files[0]) {
        mostrarErro('Por favor, selecione um arquivo PDF antes de analisar.');
        return;
      }

      const btn = document.getElementById('btn-classificar-pdf');
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> <span>Analisando PDF...</span>';

      const formData = new FormData();
      formData.append('arquivo', fileInput.files[0]);

      try {
        const resp = await fetch('/predict/pdf', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.detalhe || 'Falha ao processar o PDF');
        renderizarResultado(data, 'pdf');
      } catch (err) {
        mostrarErro('Erro: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    async function classificarTexto() {
      const texto = document.getElementById('input-texto').value.trim();
      if (texto.length < 10) {
        mostrarErro('Digite pelo menos 10 caracteres.');
        return;
      }

      const btn = document.getElementById('btn-classificar-texto');
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> <span>Classificando...</span>';

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texto: texto })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Falha ao classificar texto');
        renderizarResultado(data, 'texto');
      } catch (err) {
        mostrarErro('Erro: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    function copiarResumo() {
      if (!ultimoResultado) return;
      const r = ultimoResultado;
      const text = `Documento: ${r.classe_predita.toUpperCase()}\\nConfiança: ${r.confianca_percentual}%\\nProcessamento: ${r.latencia_ms || r.latencia_total_ms}ms\\nArquivo: ${r.arquivo || 'Texto manual'}`;
      navigator.clipboard.writeText(text).then(() => {
        mostrarToast('Resumo copiado!');
      });
    }

    function baixarJSON() {
      if (!ultimoResultado) return;
      const str = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ultimoResultado, null, 2));
      const a = document.createElement('a');
      a.setAttribute("href", str);
      a.setAttribute("download", `predicao_${ultimoResultado.classe_predita}.json`);
      document.body.appendChild(a);
      a.click();
      a.remove();
      mostrarToast('Download iniciado!');
    }

    // Healthcheck
    fetch('/health')
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          document.getElementById('nav-status-text').textContent = 'API Ativa';
          if (d.tamanho_vocabulario) {
            document.getElementById('spec-vocab').textContent = `${d.tamanho_vocabulario} termos`;
          }
        }
      })
      .catch(() => {});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

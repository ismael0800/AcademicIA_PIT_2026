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
# Interface Web de Demonstração (HTML5 + CSS Moderno + Vanilla JS)
# ==============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Interface"])
def interface_web():
    """Serve a interface gráfica interativa aprimorada para demonstração e banca examinadora."""
    html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AcademicIA — Classificador de Documentos da Pós-Graduação</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1120;
      --card-bg: #131d33;
      --card-subtle: #1a2642;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --accent-glow: rgba(56, 189, 248, 0.15);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --border: #24324f;
      --border-focus: #38bdf8;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
      --indigo: #6366f1;
      --rose: #f43f5e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      background-image: radial-gradient(circle at 50% 0%, rgba(56, 189, 248, 0.08), transparent 45%),
                        radial-gradient(circle at 10% 90%, rgba(99, 102, 241, 0.05), transparent 30%);
      color: var(--text);
      min-height: 100vh;
      padding: 2.5rem 1.25rem;
      line-height: 1.5;
    }
    .container {
      max-width: 1000px;
      margin: 0 auto;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .top-badge-row {
      display: flex;
      justify-content: center;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.3rem 0.85rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-live {
      background: rgba(16, 185, 129, 0.12);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.3);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #10b981;
      box-shadow: 0 0 8px #10b981;
      display: inline-block;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    h1 {
      font-size: 2.5rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1.05rem;
      max-width: 680px;
      margin: 0 auto 1.5rem auto;
    }

    /* Metric Bar */
    .metric-bar {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
      margin-bottom: 2rem;
    }
    .metric-pill {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      transition: transform 0.2s, border-color 0.2s;
    }
    .metric-pill:hover {
      transform: translateY(-2px);
      border-color: var(--accent);
    }
    .metric-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text);
    }
    .metric-value.highlight {
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 0.15rem;
    }

    /* Main Card */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.75rem;
      margin-bottom: 1.75rem;
      box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .tabs {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.5rem;
      padding-bottom: 0.25rem;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 0.75rem 1.25rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      border-radius: 8px 8px 0 0;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: rgba(56, 189, 248, 0.05);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    textarea {
      width: 100%;
      height: 160px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.92rem;
      resize: vertical;
      line-height: 1.6;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
    }
    .text-counter {
      display: flex;
      justify-content: flex-end;
      gap: 1rem;
      font-size: 0.75rem;
      color: var(--text-dim);
      margin-top: 0.35rem;
      margin-bottom: 1rem;
    }

    /* Dropzone */
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 12px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      background: var(--bg);
      transition: all 0.2s ease;
      margin-bottom: 1.25rem;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(56, 189, 248, 0.03);
    }
    .dropzone-icon {
      font-size: 2.75rem;
      margin-bottom: 0.75rem;
      display: inline-block;
      transition: transform 0.2s;
    }
    .dropzone:hover .dropzone-icon {
      transform: translateY(-4px) scale(1.05);
    }
    .file-selected-box {
      display: none;
      background: var(--card-subtle);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.25rem;
      margin-bottom: 1.25rem;
      align-items: center;
      justify-content: space-between;
    }
    .file-details {
      display: flex;
      align-items: center;
      gap: 0.85rem;
    }
    .file-details-icon { font-size: 1.75rem; }
    .file-name { font-weight: 600; font-size: 0.9rem; color: var(--text); }
    .file-size { font-size: 0.75rem; color: var(--text-muted); }
    .btn-remove {
      background: none;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      padding: 0.4rem;
      font-size: 1.1rem;
      border-radius: 6px;
      transition: color 0.2s, background 0.2s;
    }
    .btn-remove:hover {
      color: var(--danger);
      background: rgba(239, 68, 68, 0.1);
    }

    /* Buttons */
    .btn {
      background: var(--accent);
      color: #0b1120;
      border: none;
      padding: 0.8rem 1.75rem;
      border-radius: 10px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      box-shadow: 0 4px 14px rgba(56, 189, 248, 0.3);
    }
    .btn:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
      box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
    }
    .btn:active { transform: translateY(0); }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    .btn-outline {
      background: transparent;
      color: var(--text-muted);
      border: 1px solid var(--border);
      box-shadow: none;
    }
    .btn-outline:hover {
      background: var(--card-subtle);
      color: var(--text);
      border-color: var(--border-focus);
      box-shadow: none;
    }

    /* Samples Strip */
    .samples-section {
      margin-top: 1.5rem;
      padding-top: 1.25rem;
      border-top: 1px solid var(--border);
    }
    .samples-header {
      font-size: 0.8rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .samples-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .sample-chip {
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.4rem 0.8rem;
      border-radius: 8px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .sample-chip:hover {
      border-color: var(--accent);
      color: var(--text);
      background: rgba(56, 189, 248, 0.08);
      transform: translateY(-1px);
    }

    /* Loading Spinner */
    .spinner {
      display: inline-block;
      width: 18px;
      height: 18px;
      border: 2px solid rgba(11, 17, 32, 0.25);
      border-radius: 50%;
      border-top-color: #0b1120;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    /* Result Card */
    #resultado-card {
      display: none;
      animation: slideUp 0.35s ease-out;
      border: 1px solid rgba(56, 189, 248, 0.3);
      position: relative;
      overflow: hidden;
    }
    #resultado-card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 4px;
      background: linear-gradient(90deg, #38bdf8, #818cf8, #a855f7);
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .res-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .res-title-block {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .res-kicker {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 600;
    }
    .class-tag {
      font-size: 1.4rem;
      font-weight: 800;
      padding: 0.45rem 1.25rem;
      border-radius: 10px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      width: fit-content;
    }
    .tag-edital { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; }
    .tag-resultado { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }
    .tag-homologacao { background: rgba(168, 85, 247, 0.15); color: #a855f7; border: 1px solid #a855f7; }
    .tag-recurso { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; }
    .tag-aviso { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid #818cf8; }
    .tag-retificacao { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid #f43f5e; }
    .tag-indeterminado { background: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid #94a3b8; }

    .confidence-badge {
      display: flex;
      flex-direction: column;
      align-items: flex-end;
    }
    .confidence-number {
      font-size: 2.25rem;
      font-weight: 800;
      line-height: 1;
      color: #38bdf8;
    }
    .confidence-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-top: 0.25rem;
    }

    /* Result Meta Grid */
    .res-meta-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 1.75rem;
    }
    .res-meta-item {
      display: flex;
      flex-direction: column;
    }
    .res-meta-title {
      font-size: 0.7rem;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .res-meta-val {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text);
      margin-top: 0.15rem;
    }

    /* Distribution Bars */
    .dist-section-title {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 1rem;
      display: flex;
      justify-content: space-between;
    }
    .dist-grid {
      display: grid;
      gap: 0.7rem;
      margin-bottom: 1.75rem;
    }
    .dist-row {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.65rem 0.85rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    .dist-row.winner {
      border-color: rgba(56, 189, 248, 0.4);
      background: rgba(56, 189, 248, 0.03);
    }
    .dist-labels {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }
    .dist-class-name {
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .progress-track {
      background: #1a2642;
      border-radius: 9999px;
      height: 8px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      border-radius: 9999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Snippet Box */
    .snippet-box {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      color: #cbd5e1;
      line-height: 1.6;
      max-height: 140px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
      margin-bottom: 1.5rem;
    }

    /* Actions Footer */
    .res-actions {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      justify-content: flex-end;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }

    /* Alert */
    .alert {
      display: none;
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #fca5a5;
      padding: 1rem 1.25rem;
      border-radius: 10px;
      margin-bottom: 1.5rem;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    /* Toast */
    .toast {
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      background: var(--card-bg);
      border: 1px solid var(--accent);
      color: var(--text);
      padding: 0.75rem 1.25rem;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 500;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.3s ease;
      pointer-events: none;
      z-index: 100;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }

    footer {
      text-align: center;
      margin-top: 3rem;
      font-size: 0.85rem;
      color: var(--text-dim);
    }
    footer a {
      color: var(--accent);
      text-decoration: none;
      transition: color 0.2s;
    }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="top-badge-row">
        <div class="badge">Projeto Integrador de Nicho do Zero • Piauí</div>
        <div class="badge badge-live">
          <span class="status-dot"></span>
          <span id="api-status-text">PyTorch CPU Ativo • best.pt</span>
        </div>
      </div>
      <h1>AcademicIA</h1>
      <p class="subtitle">
        Triagem e Classificação Inteligente de Editais, Resultados, Recursos e Homologações da Pós-Graduação Stricto Sensu (UFPI, UESPI, IFPI e UFDPAR).
      </p>

      <!-- Metric Strip -->
      <div class="metric-bar">
        <div class="metric-pill">
          <span class="metric-value highlight">78.3%</span>
          <span class="metric-label">Acurácia de Teste</span>
        </div>
        <div class="metric-pill">
          <span class="metric-value">0.7899</span>
          <span class="metric-label">F1-Score Ponderado</span>
        </div>
        <div class="metric-pill">
          <span class="metric-value">&lt; 10ms</span>
          <span class="metric-label">Latência em CPU</span>
        </div>
        <div class="metric-pill">
          <span class="metric-value">324</span>
          <span class="metric-label">Corpus de PDFs</span>
        </div>
        <div class="metric-pill">
          <span class="metric-value">6</span>
          <span class="metric-label">Classes Oficiais</span>
        </div>
      </div>
    </header>

    <!-- Error Banner -->
    <div id="error-alert" class="alert" style="display: none;">
      <span id="error-msg">Erro ao processar documento.</span>
      <button onclick="fecharAlerta()" style="background:none;border:none;color:#fca5a5;cursor:pointer;font-size:1.1rem;">✕</button>
    </div>

    <!-- Main Input Card -->
    <div class="card">
      <div class="tabs">
        <button class="tab-btn active" id="btn-tab-texto" onclick="switchTab('tab-texto')">
          📝 Digitar ou Colar Texto
        </button>
        <button class="tab-btn" id="btn-tab-pdf" onclick="switchTab('tab-pdf')">
          📄 Upload de Arquivo PDF
        </button>
      </div>

      <!-- Aba 1: Texto -->
      <div id="tab-texto" class="tab-content active">
        <textarea id="input-texto" placeholder="Cole aqui o texto da primeira página do edital, resultado, recurso, homologação ou retificação..." oninput="atualizarContadorTexto()"></textarea>
        <div class="text-counter">
          <span id="char-count">0 caracteres</span>
          <span>•</span>
          <span id="word-count">0 palavras</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <button class="btn" id="btn-classificar-texto" onclick="classificarTexto()">
            <span>⚡ Classificar Documento</span>
          </button>
          <button class="btn btn-outline" onclick="limparTexto()">Limpar</button>
        </div>

        <!-- 6 Exemplos Oficiais -->
        <div class="samples-section">
          <div class="samples-header">
            <span>✨ Exemplos rápidos de teste (1 clique para carregar):</span>
          </div>
          <div class="samples-grid">
            <button class="sample-chip" onclick="carregarExemplo('edital')">📜 Edital 2026</button>
            <button class="sample-chip" onclick="carregarExemplo('homologacao')">📋 Homologação</button>
            <button class="sample-chip" onclick="carregarExemplo('resultado')">📊 Resultado Final</button>
            <button class="sample-chip" onclick="carregarExemplo('recurso')">⚖️ Interposição Recurso</button>
            <button class="sample-chip" onclick="carregarExemplo('aviso')">📢 Aviso de Convocação</button>
            <button class="sample-chip" onclick="carregarExemplo('retificacao')">✏️ Retificação / Errata</button>
          </div>
        </div>
      </div>

      <!-- Aba 2: PDF -->
      <div id="tab-pdf" class="tab-content">
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept="application/pdf" style="display:none" onchange="tratarArquivoSelecionado(this)">
          <div class="dropzone-icon">📥</div>
          <p style="font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem;">
            Clique ou arraste um arquivo PDF aqui
          </p>
          <span style="font-size: 0.85rem; color: var(--text-muted); display:block;">
            O modelo extrai a primeira página e classifica em menos de 10 milissegundos
          </span>
        </div>

        <!-- Card de Arquivo Selecionado -->
        <div class="file-selected-box" id="file-box">
          <div class="file-details">
            <span class="file-details-icon">📄</span>
            <div>
              <div class="file-name" id="file-name-label">documento.pdf</div>
              <div class="file-size" id="file-size-label">0 KB</div>
            </div>
          </div>
          <button class="btn-remove" title="Remover arquivo" onclick="removerArquivo()">✕</button>
        </div>

        <button class="btn" id="btn-classificar-pdf" onclick="classificarPDF()">
          <span>⚡ Analisar Arquivo PDF</span>
        </button>
      </div>
    </div>

    <!-- Card de Resultado -->
    <div id="resultado-card" class="card">
      <div class="res-header">
        <div class="res-title-block">
          <span class="res-kicker">Classificação Predita pela Rede Neural:</span>
          <div id="res-tag" class="class-tag tag-edital">
            <span id="res-icon">📜</span>
            <span id="res-label">EDITAL</span>
          </div>
        </div>
        <div class="confidence-badge">
          <div class="confidence-number" id="res-conf">98.5%</div>
          <div class="confidence-label">Nível de Certeza</div>
        </div>
      </div>

      <!-- Metadados da Inferência -->
      <div class="res-meta-grid">
        <div class="res-meta-item">
          <span class="res-meta-title">Latência Neural</span>
          <span class="res-meta-val" id="res-lat">8.2 ms</span>
        </div>
        <div class="res-meta-item">
          <span class="res-meta-title">Tokens Analisados</span>
          <span class="res-meta-val" id="res-tokens">142 tokens</span>
        </div>
        <div class="res-meta-item">
          <span class="res-meta-title">Páginas Lidas</span>
          <span class="res-meta-val" id="res-paginas">1 página</span>
        </div>
        <div class="res-meta-item">
          <span class="res-meta-title">Origem</span>
          <span class="res-meta-val" id="res-origem">Texto Direto</span>
        </div>
      </div>

      <!-- Distribuição de Probabilidades -->
      <div class="dist-section-title">
        <span>Distribuição de Probabilidades (6 Classes Softmax):</span>
        <span style="font-size: 0.75rem; color: var(--text-dim);">Ordenado por Maior Afinidade</span>
      </div>
      <div class="dist-grid" id="dist-container"></div>

      <!-- Trecho Inspecionado -->
      <div style="margin-bottom: 0.5rem;">
        <span class="dist-section-title" style="margin-bottom: 0.4rem;">Texto Efetivo Analisado pela Camada de Embedding:</span>
        <div class="snippet-box" id="res-trecho">Aguardando texto...</div>
      </div>

      <!-- Ações do Resultado -->
      <div class="res-actions">
        <button class="btn btn-outline" onclick="copiarResumo()">
          📋 Copiar Resumo
        </button>
        <button class="btn btn-outline" onclick="baixarJSON()">
          📥 Baixar Relatório JSON
        </button>
        <button class="btn btn-outline" onclick="fecharResultado()">
          ✕ Ocultar
        </button>
      </div>
    </div>

    <!-- Toast Notification -->
    <div id="toast" class="toast">Resumo copiado para a área de transferência!</div>

    <footer>
      AcademicIA 2026 • Pesos Próprios Inicializados do Zero • PyTorch + FastAPI • 
      <a href="/docs" target="_blank">Swagger /docs</a> • 
      <a href="/redoc" target="_blank">ReDoc</a>
    </footer>
  </div>

  <script>
    // 6 Exemplos Reais representativos do Stricto Sensu
    const EXEMPLOS = {
      edital: "UNIVERSIDADE FEDERAL DO PIAUÍ - UFPI\\nPRÓ-REITORIA DE ENSINO DE PÓS-GRADUAÇÃO\\nEDITAL Nº 01/2026 - SELEÇÃO PÚBLICA PARA O CURSO DE MESTRADO EM COMUNICAÇÃO\\nO Diretor do Centro de Ciências da Educação, no uso de suas atribuições legais, torna pública a abertura de inscrições para o processo seletivo de candidatos ao Programa de Pós-Graduação em Comunicação (PPGCOM/UFPI), com oferta de 15 vagas regulares para ingresso no primeiro semestre letivo.",
      homologacao: "PROGRAMA DE PÓS-GRADUAÇÃO EM ESTUDOS LITERÁRIOS - PPGEL/UFPI\\nHOMOLOGAÇÃO DAS INSCRIÇÕES - PROCESSO SELETIVO TURMA 2026\\nA Comissão de Seleção torna pública a homologação das inscrições deferidas e indeferidas dos candidatos inscritos no Edital nº 02/2026 para o curso de Mestrado, após a análise documental e tempestividade das submissões.",
      resultado: "COORDENAÇÃO DO PROGRAMA DE PÓS-GRADUAÇÃO EM CIÊNCIA DOS MATERIAIS\\nDIVULGAÇÃO DO RESULTADO FINAL DA PROVA ESCRITA E ANÁLISE DE CURRÍCULO\\nA Comissão de Seleção torna pública a lista final de candidatos aprovados e classificados por ordem decrescente de pontuação no processo seletivo 2026, com suas respectivas notas finais.",
      recurso: "ILUSTRÍSSIMO SENHOR PRESIDENTE DA COMISSÃO DE SELEÇÃO DO PPGCOM\\nFORMULÁRIO DE INTERPOSIÇÃO DE RECURSO ADMINISTRATIVO CONTRA O RESULTADO PRELIMINAR\\nO candidato abaixo assinado vem respeitosamente perante Vossa Senhoria interpor RECURSO contra a pontuação atribuída na etapa de avaliação do projeto de pesquisa, solicitando nova aferição conforme fundamentos em anexo.",
      aviso: "AVISO DE CONVOCAÇÃO PARA MATRÍCULA INSTITUCIONAL - MESTRADO E DOUTORADO\\nA Coordenação do Programa de Pós-Graduação convoca todos os candidatos aprovados em 1ª chamada no Edital nº 01/2026 a comparecerem à secretaria integrada para entrega dos documentos comprobatórios e realização da matrícula no período estipulado no cronograma.",
      retificacao: "RETIFICAÇÃO Nº 01 AO EDITAL Nº 03/2026 - PROCESSO SELETIVO PPGEDUCAÇÃO\\nA Direção do Centro torna pública a alteração do cronograma de seleção. Onde se lê: 'Data limite para envio do projeto: 15 de março', leia-se: 'Data limite para envio do projeto: 22 de março'. Ficam inalteradas e ratificadas as demais disposições do edital original."
    };

    const ICONES_CLASSES = {
      edital: "📜",
      homologacao: "📋",
      resultado: "📊",
      recurso: "⚖️",
      aviso: "📢",
      retificacao: "✏️",
      indeterminado: "❓"
    };

    const CORES_CLASSES = {
      edital: "#38bdf8",
      homologacao: "#a855f7",
      resultado: "#10b981",
      recurso: "#f59e0b",
      aviso: "#818cf8",
      retificacao: "#f43f5e",
      indeterminado: "#94a3b8"
    };

    let ultimoResultado = null;

    // Tabs
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      if (tabId === 'tab-texto') {
        document.getElementById('btn-tab-texto').classList.add('active');
      } else {
        document.getElementById('btn-tab-pdf').classList.add('active');
      }
      document.getElementById(tabId).classList.add('active');
    }

    // Text Counter
    function atualizarContadorTexto() {
      const val = document.getElementById('input-texto').value;
      document.getElementById('char-count').textContent = val.length + ' caracteres';
      const words = val.trim() ? val.trim().split(/\\s+/).length : 0;
      document.getElementById('word-count').textContent = words + ' palavras';
    }

    function limparTexto() {
      document.getElementById('input-texto').value = '';
      atualizarContadorTexto();
    }

    function carregarExemplo(tipo) {
      if (EXEMPLOS[tipo]) {
        document.getElementById('input-texto').value = EXEMPLOS[tipo];
        atualizarContadorTexto();
        switchTab('tab-texto');
      }
    }

    // Drag and Drop
    const dropzone = document.getElementById('dropzone');
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      }, false);
    });
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0 && files[0].name.toLowerCase().endsWith('.pdf')) {
        document.getElementById('file-input').files = files;
        tratarArquivoSelecionado(document.getElementById('file-input'));
      } else {
        mostrarErro('Por favor, arraste um arquivo com extensão .pdf válido.');
      }
    }, false);

    function tratarArquivoSelecionado(input) {
      if (input.files && input.files[0]) {
        const file = input.files[0];
        document.getElementById('file-name-label').textContent = file.name;
        const sizeKb = (file.size / 1024).toFixed(1);
        const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
        document.getElementById('file-size-label').textContent = file.size > 1048576 ? `${sizeMb} MB` : `${sizeKb} KB`;
        
        document.getElementById('dropzone').style.display = 'none';
        document.getElementById('file-box').style.display = 'flex';
      }
    }

    function removerArquivo() {
      document.getElementById('file-input').value = '';
      document.getElementById('dropzone').style.display = 'block';
      document.getElementById('file-box').style.display = 'none';
    }

    // Alertas e Toast
    function mostrarErro(msg) {
      const el = document.getElementById('error-alert');
      document.getElementById('error-msg').textContent = msg;
      el.style.display = 'flex';
    }

    function fecharAlerta() {
      document.getElementById('error-alert').style.display = 'none';
    }

    function mostrarToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2800);
    }

    function fecharResultado() {
      document.getElementById('resultado-card').style.display = 'none';
    }

    // Renderizador de Resultados
    function renderizarResultado(data, origem) {
      ultimoResultado = data;
      fecharAlerta();

      const card = document.getElementById('resultado-card');
      const tag = document.getElementById('res-tag');
      const classe = data.classe_predita || 'indeterminado';

      tag.className = 'class-tag tag-' + classe;
      document.getElementById('res-icon').textContent = ICONES_CLASSES[classe] || '📄';
      document.getElementById('res-label').textContent = classe.toUpperCase();

      const conf = (data.confianca_percentual || 0).toFixed(1) + '%';
      document.getElementById('res-conf').textContent = conf;
      document.getElementById('res-conf').style.color = CORES_CLASSES[classe] || '#38bdf8';

      const lat = (data.latencia_ms || data.latencia_total_ms || 0).toFixed(1) + ' ms';
      document.getElementById('res-lat').textContent = lat;

      document.getElementById('res-tokens').textContent = (data.tokens_processados || 0) + ' tokens';
      document.getElementById('res-paginas').textContent = (data.paginas_totais || 1) + ' pág(s)';
      document.getElementById('res-origem').textContent = data.arquivo ? `PDF: ${data.arquivo}` : 'Texto Colado';

      document.getElementById('res-trecho').textContent = data.trecho_analisado || data.trecho_inicial || 'Sem trecho extraído.';

      // Barras de Probabilidade ordenadas
      const dist = data.distribuicao_probabilidades || {};
      const sortedEntries = Object.entries(dist).sort((a, b) => b[1] - a[1]);
      const distContainer = document.getElementById('dist-container');
      distContainer.innerHTML = '';

      sortedEntries.forEach(([cls, pct]) => {
        const isWinner = (cls === classe);
        const cor = CORES_CLASSES[cls] || '#38bdf8';
        const icone = ICONES_CLASSES[cls] || '•';

        distContainer.innerHTML += `
          <div class="dist-row ${isWinner ? 'winner' : ''}">
            <div class="dist-labels">
              <span class="dist-class-name" style="color: ${isWinner ? cor : 'var(--text)'};">
                <span>${icone}</span>
                <span>${cls.toUpperCase()}</span>
                ${isWinner ? '<span style="font-size: 0.7rem; background: rgba(56, 189, 248, 0.2); padding: 0.1rem 0.4rem; border-radius: 4px; margin-left: 0.4rem;">Vencedor</span>' : ''}
              </span>
              <strong style="color: ${isWinner ? cor : 'var(--text-muted)'};">${pct}%</strong>
            </div>
            <div class="progress-track">
              <div class="progress-bar" style="width: ${pct}%; background: ${cor};"></div>
            </div>
          </div>
        `;
      });

      card.style.display = 'block';
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Inferência Texto
    async function classificarTexto() {
      const texto = document.getElementById('input-texto').value.trim();
      if (texto.length < 10) {
        mostrarErro('Por favor, digite ou cole pelo menos 10 caracteres para inferência.');
        return;
      }

      const btn = document.getElementById('btn-classificar-texto');
      const originalHTML = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> <span>Inferindo com PyTorch...</span>';

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texto: texto })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Falha ao processar texto.');
        renderizarResultado(data, 'texto');
      } catch (err) {
        mostrarErro('Erro na inferência: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
      }
    }

    // Inferência PDF
    async function classificarPDF() {
      const fileInput = document.getElementById('file-input');
      if (!fileInput.files || !fileInput.files[0]) {
        mostrarErro('Por favor, selecione ou arraste um arquivo PDF antes de classificar.');
        return;
      }

      const btn = document.getElementById('btn-classificar-pdf');
      const originalHTML = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> <span>Lendo PDF e inferindo...</span>';

      const formData = new FormData();
      formData.append('arquivo', fileInput.files[0]);

      try {
        const resp = await fetch('/predict/pdf', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.detalhe || 'Erro ao processar PDF');
        renderizarResultado(data, 'pdf');
      } catch (err) {
        mostrarErro('Falha no upload/inferência do PDF: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalHTML;
      }
    }

    // Ações do Resultado
    function copiarResumo() {
      if (!ultimoResultado) return;
      const r = ultimoResultado;
      const texto = `[AcademicIA Predição]\\nDocumento: ${r.classe_predita ? r.classe_predita.toUpperCase() : 'N/A'}\\nConfiança: ${r.confianca_percentual}%\\nLatência: ${r.latencia_ms || r.latencia_total_ms} ms\\nTokens: ${r.tokens_processados}\\nArquivo: ${r.arquivo || 'Texto manual'}`;
      
      navigator.clipboard.writeText(texto).then(() => {
        mostrarToast('Resumo copiado para a área de transferência!');
      }).catch(() => {
        mostrarToast('Falha ao copiar.');
      });
    }

    function baixarJSON() {
      if (!ultimoResultado) return;
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(ultimoResultado, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `predicao_${ultimoResultado.classe_predita || 'resultado'}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      mostrarToast('Download do JSON iniciado!');
    }

    // Healthcheck inicial para atualizar status
    fetch('/health')
      .then(r => r.json())
      .then(d => {
        if (d.status === 'ok') {
          document.getElementById('api-status-text').textContent = `PyTorch ${d.dispositivo.toUpperCase()} • ${d.modelo} (${d.tamanho_vocabulario} termos)`;
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

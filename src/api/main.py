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
    version="1.0.0",
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
            versao="1.0.0",
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
    """Serve a interface gráfica interativa para demonstração e banca examinadora."""
    html = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AcademicIA — Classificador de Documentos da Pós-Graduação</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #334155;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container {
      max-width: 960px;
      margin: 0 auto;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.3);
      margin-bottom: 0.75rem;
    }
    h1 {
      font-size: 2.25rem;
      font-weight: 700;
      background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 1rem;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .tabs {
      display: flex;
      gap: 0.5rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1.25rem;
    }
    .tab-btn {
      background: none;
      border: none;
      color: var(--text-muted);
      padding: 0.75rem 1rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: all 0.2s;
    }
    .tab-btn.active {
      color: var(--accent);
      border-bottom-color: var(--accent);
    }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    textarea {
      width: 100%;
      height: 140px;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem;
      color: var(--text);
      font-family: inherit;
      font-size: 0.9rem;
      resize: vertical;
      margin-bottom: 1rem;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
    }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 2rem 1rem;
      text-align: center;
      cursor: pointer;
      background: #0f172a;
      transition: border-color 0.2s;
      margin-bottom: 1rem;
    }
    .dropzone:hover { border-color: var(--accent); }
    .btn {
      background: var(--accent);
      color: #0f172a;
      border: none;
      padding: 0.75rem 1.5rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.95rem;
      cursor: pointer;
      transition: background 0.2s;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .btn:hover { background: var(--accent-hover); }
    .btn-secondary {
      background: #334155;
      color: var(--text);
    }
    .btn-secondary:hover { background: #475569; }
    .samples {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.75rem;
      align-items: center;
    }
    .samples span { font-size: 0.8rem; color: var(--text-muted); }
    .sample-chip {
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .sample-chip:hover {
      border-color: var(--accent);
      color: var(--text);
    }
    #resultado-card {
      display: none;
      animation: fadeIn 0.3s ease-in-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .res-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }
    .class-tag {
      font-size: 1.25rem;
      font-weight: 700;
      padding: 0.4rem 1rem;
      border-radius: 8px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .tag-edital { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }
    .tag-resultado { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .tag-homologacao { background: rgba(168, 85, 247, 0.2); color: #a855f7; border: 1px solid #a855f7; }
    .tag-recurso { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .tag-aviso { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #60a5fa; }
    .tag-retificacao { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #f87171; }
    .tag-indeterminado { background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid #94a3b8; }
    .meta-metrics {
      display: flex;
      gap: 1.5rem;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .dist-bar {
      margin-bottom: 0.75rem;
    }
    .dist-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      margin-bottom: 0.25rem;
    }
    .progress-bg {
      background: #0f172a;
      border-radius: 9999px;
      height: 8px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 9999px;
      transition: width 0.4s ease;
    }
    footer {
      text-align: center;
      margin-top: 2rem;
      font-size: 0.8rem;
      color: var(--text-muted);
    }
    footer a { color: var(--accent); text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">Projeto Integrador de Nicho do Zero • Piauí</div>
      <h1>AcademicIA</h1>
      <p class="subtitle">Classificação Automatizada de Editais e Documentos da Pós-Graduação (Deep Learning em PyTorch)</p>
    </header>

    <div class="card">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-texto')">📝 Digitar ou Colar Texto</button>
        <button class="tab-btn" onclick="switchTab('tab-pdf')">📄 Enviar Arquivo PDF</button>
      </div>

      <!-- Aba Texto -->
      <div id="tab-texto" class="tab-content active">
        <textarea id="input-texto" placeholder="Cole aqui o texto da primeira página do edital, resultado, recurso ou homologação..."></textarea>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
          <button class="btn" onclick="classificarTexto()">⚡ Classificar Documento</button>
          <div class="samples">
            <span>Exemplos rápidos:</span>
            <button class="sample-chip" onclick="carregarExemplo('edital')">Edital 2026</button>
            <button class="sample-chip" onclick="carregarExemplo('resultado')">Resultado Final</button>
            <button class="sample-chip" onclick="carregarExemplo('recurso')">Interposição Recurso</button>
            <button class="sample-chip" onclick="carregarExemplo('homologacao')">Homologação</button>
          </div>
        </div>
      </div>

      <!-- Aba PDF -->
      <div id="tab-pdf" class="tab-content">
        <div class="dropzone" onclick="document.getElementById('file-input').click()">
          <input type="file" id="file-input" accept="application/pdf" style="display:none" onchange="mostrarArquivoSelecionado(this)">
          <div style="font-size: 2rem; margin-bottom: 0.5rem;">📄</div>
          <p id="file-label" style="font-size: 0.95rem; color: var(--text-muted);">Clique ou arraste um arquivo PDF aqui</p>
          <span style="font-size: 0.75rem; color: var(--text-muted); display:block; margin-top:0.25rem;">Lê automaticamente a 1ª página e roda a inferência</span>
        </div>
        <button class="btn" onclick="classificarPDF()">⚡ Analisar Arquivo PDF</button>
      </div>
    </div>

    <!-- Card de Resultado -->
    <div id="resultado-card" class="card">
      <div class="res-header">
        <div>
          <span style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-bottom: 0.25rem;">PREDIÇÃO DO MODELO PYTORCH:</span>
          <span id="res-tag" class="class-tag tag-edital">EDITAL</span>
        </div>
        <div class="meta-metrics">
          <div><strong id="res-conf">0%</strong> de confiança</div>
          <div>Latência: <strong id="res-lat">0ms</strong></div>
        </div>
      </div>

      <h3 style="font-size: 0.95rem; margin-bottom: 1rem; color: var(--text-muted);">Distribuição de Probabilidades (6 Classes):</h3>
      <div id="dist-container"></div>

      <div style="margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--text-muted);">
        <strong>Trecho analisado pela rede:</strong>
        <p id="res-trecho" style="margin-top: 0.25rem; font-style: italic;"></p>
      </div>
    </div>

    <footer>
      AcademicIA 2026 • Pesos Próprios Inicializados do Zero • <a href="/docs" target="_blank">Documentação da API (Swagger /docs)</a>
    </footer>
  </div>

  <script>
    const EXEMPLOS = {
      edital: "UNIVERSIDADE FEDERAL DO PIAUÍ - UFPI\\nPRÓ-REITORIA DE ENSINO DE PÓS-GRADUAÇÃO\\nEDITAL Nº 01/2026 - SELEÇÃO PÚBLICA PARA O CURSO DE MESTRADO EM COMUNICAÇÃO\\nO Diretor do Centro no uso de suas atribuições torna pública a abertura de inscrições para o processo seletivo de candidatos ao Programa de Pós-Graduação.",
      resultado: "COORDENAÇÃO DO PROGRAMA DE PÓS-GRADUAÇÃO EM CIÊNCIA DOS MATERIAIS\\nDIVULGAÇÃO DO RESULTADO FINAL DA PROVA ESCRITA E ANÁLISE DE CURRÍCULO\\nA Comissão de Seleção torna pública a lista final de candidatos aprovados e classificados por ordem decrescente de pontuação.",
      recurso: "ILUSTRÍSSIMO SENHOR PRESIDENTE DA COMISSÃO DE SELEÇÃO\\nFORMULÁRIO DE INTERPOSIÇÃO DE RECURSO ADMINISTRATIVO CONTRA O RESULTADO PRELIMINAR\\nO candidato vem respeitosamente requerer a revisão da nota atribuída à prova escrita com base nas justificativas anexas.",
      homologacao: "HOMOLOGAÇÃO DAS INSCRIÇÕES - PROCESSO SELETIVO TURMA 2026\\nA Coordenação do Programa torna pública a relação definitiva de candidatos com inscrições homologadas após análise da documentação exigida no edital."
    };

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    function carregarExemplo(tipo) {
      document.getElementById('input-texto').value = EXEMPLOS[tipo];
    }

    function mostrarArquivoSelecionado(input) {
      if (input.files && input.files[0]) {
        document.getElementById('file-label').textContent = 'Selecionado: ' + input.files[0].name;
      }
    }

    function renderizarResultado(data) {
      const card = document.getElementById('resultado-card');
      const tag = document.getElementById('res-tag');
      const classe = data.classe_predita || 'indeterminado';

      tag.textContent = classe.toUpperCase();
      tag.className = 'class-tag tag-' + classe;

      document.getElementById('res-conf').textContent = (data.confianca_percentual || 0) + '%';
      document.getElementById('res-lat').textContent = (data.latencia_ms || data.latencia_total_ms || 0) + 'ms';
      document.getElementById('res-trecho').textContent = data.trecho_analisado || data.trecho_inicial || 'Nenhum texto extraído.';

      const dist = data.distribuicao_probabilidades || {};
      const distContainer = document.getElementById('dist-container');
      distContainer.innerHTML = '';

      for (const [cls, pct] of Object.entries(dist)) {
        distContainer.innerHTML += `
          <div class="dist-bar">
            <div class="dist-labels">
              <span>${cls.toUpperCase()}</span>
              <strong>${pct}%</strong>
            </div>
            <div class="progress-bg">
              <div class="progress-fill" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }

      card.style.display = 'block';
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async function classificarTexto() {
      const texto = document.getElementById('input-texto').value.trim();
      if (texto.length < 10) {
        alert('Por favor, digite ou cole pelo menos 10 caracteres.');
        return;
      }

      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ texto: texto })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Erro na requisição');
        renderizarResultado(data);
      } catch (err) {
        alert('Erro ao classificar: ' + err.message);
      }
    }

    async function classificarPDF() {
      const fileInput = document.getElementById('file-input');
      if (!fileInput.files || !fileInput.files[0]) {
        alert('Por favor, selecione um arquivo PDF primeiro.');
        return;
      }

      const formData = new FormData();
      formData.append('arquivo', fileInput.files[0]);

      try {
        const resp = await fetch('/predict/pdf', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.detalhe || 'Erro ao processar PDF');
        renderizarResultado(data);
      } catch (err) {
        alert('Erro ao classificar PDF: ' + err.message);
      }
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

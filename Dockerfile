# ==============================================================================
# Dockerfile de Produção para Inferência - AcademicIA
# ==============================================================================
FROM python:3.12-slim

# Evita criação de arquivos .pyc e faz flush imediato do stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Instala dependências de sistema mínimas se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python em camada de cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código-fonte, modelo treinado e relatórios de avaliação
COPY src ./src
COPY models/best.pt ./models/best.pt
COPY models/vocab.json ./models/vocab.json
COPY reports ./reports
COPY params.yaml .

# Cria usuário não-root para segurança em produção
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check nativo do Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização do servidor Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

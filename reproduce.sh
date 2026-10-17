#!/usr/bin/env bash
# ==============================================================================
# Script de Reprodução Automatizada - AcademicIA (Linux / macOS / Bash)
# ==============================================================================
set -e

echo "================================================================="
echo "   AUDITORIA E TESTE DE REPRODUÇÃO AUTOMATIZADA - AcademicIA    "
echo "================================================================="

# 1. Detecta o interpretador Python
PYTHON_BIN="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
    echo "[1/4] Usando ambiente virtual: $PYTHON_BIN"
elif [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
    echo "[1/4] Usando ambiente virtual: $PYTHON_BIN"
else
    echo "[1/4] Usando interpretador do sistema: $PYTHON_BIN"
fi

$PYTHON_BIN --version

# 2. Executa a suíte de testes
echo -e "\n[2/4] Executando testes automatizados com pytest..."
$PYTHON_BIN -m pytest tests/ -v

# 3. Teste de inferência
echo -e "\n[3/4] Testando motor de inferência em produção (models/best.pt)..."
$PYTHON_BIN src/models/inference.py --texto "EDITAL DE ABERTURA Nº 01/2026 - A Coordenação do Programa de Pós-Graduação torna pública a abertura do processo seletivo para ingresso no mestrado acadêmico."

# 4. Checagem de artefatos
echo -e "\n[4/4] Verificando integridade dos artefatos críticos..."
ARQUIVOS=(
    "models/best.pt"
    "models/vocab.json"
    "reports/figures/training_curves.png"
    "reports/figures/confusion_matrix.png"
    "reports/metrics/error_analysis.json"
    "Dockerfile"
    "requirements.txt"
)

for arq in "${ARQUIVOS[@]}"; do
    if [ -f "$arq" ]; then
        echo "  [OK] $arq"
    else
        echo "  [FALHA] $arq não encontrado!"
        exit 1
    fi
done

echo -e "\n================================================================="
echo "🎉 REPRODUÇÃO CONCLUÍDA COM 100% DE SUCESSO!"
echo "Para iniciar a API:"
echo "  $PYTHON_BIN -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo "================================================================="

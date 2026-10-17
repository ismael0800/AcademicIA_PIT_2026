# ==============================================================================
# Script de Reprodução Automatizada - AcademicIA (Windows PowerShell)
# ==============================================================================
# Executa a auditoria completa de ponta a ponta:
# 1. Validação de ambiente e dependências
# 2. Execução da suíte de testes unitários (pytest)
# 3. Teste de inferência do modelo com checkpoint (best.pt)
# 4. Verificação de integridade dos artefatos
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   AUDITORIA E TESTE DE REPRODUÇÃO AUTOMATIZADA - AcademicIA    " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Detecta o interpretador Python
$PythonExe = "python"
if (Test-Path ".\.venv\Scripts\python.exe") {
    $PythonExe = ".\.venv\Scripts\python.exe"
    Write-Host "[1/4] Usando ambiente virtual local: $PythonExe" -ForegroundColor Green
} else {
    Write-Host "[1/4] Usando Python do sistema: $PythonExe" -ForegroundColor Yellow
}

# 2. Verifica a versão do Python
& $PythonExe --version

# 3. Execução da Suíte de Testes Unitários
Write-Host "`n[2/4] Executando suíte de testes automatizados com PyTest..." -ForegroundColor Cyan
& $PythonExe -m pytest tests/ -v
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Falha nos testes automatizados!" -ForegroundColor Red
    exit 1
}
Write-Host "  -> Todos os 8 testes unitários foram APROVADOS!" -ForegroundColor Green

# 4. Teste de Inferência do Modelo Checkpoint
Write-Host "`n[3/4] Testando motor de inferência em produção (models/best.pt)..." -ForegroundColor Cyan
& $PythonExe src/models/inference.py --texto "EDITAL DE ABERTURA Nº 01/2026 - A Coordenação do Programa de Pós-Graduação torna pública a abertura do processo seletivo para ingresso no mestrado acadêmico."
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Falha no motor de inferência!" -ForegroundColor Red
    exit 1
}
Write-Host "  -> Inferência em tempo real executada com sucesso!" -ForegroundColor Green

# 5. Auditoria de Artefatos MLOps
Write-Host "`n[4/4] Verificando integridade dos artefatos do projeto..." -ForegroundColor Cyan
$ArquivosCriticos = @(
    "models/best.pt",
    "models/vocab.json",
    "reports/figures/training_curves.png",
    "reports/figures/confusion_matrix.png",
    "reports/metrics/error_analysis.json",
    "Dockerfile",
    "requirements.txt"
)

foreach ($arq in $ArquivosCriticos) {
    if (Test-Path $arq) {
        Write-Host "  [OK] $arq" -ForegroundColor Green
    } else {
        Write-Host "  [FALHA] $arq não encontrado!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
Write-Host "🎉 REPRODUÇÃO CONCLUÍDA COM 100% DE SUCESSO!" -ForegroundColor Green
Write-Host "Para iniciar o servidor web da API e interface:" -ForegroundColor White
Write-Host "  & $PythonExe -m uvicorn src.api.main:app --port 8000 --reload" -ForegroundColor Yellow
Write-Host "E acesse: http://localhost:8000" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan

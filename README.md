# AcademicIA — Classificador Inteligente de Documentos de Pós-Graduação

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20CPU--First-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version-945DD6.svg)](https://dvc.org/)
[![Tests](https://img.shields.io/badge/PyTest-8%20Passed%20(100%25)-brightgreen.svg)]()

> **Projeto Integrador de Nicho do Zero · Engenharia de Software e Inteligência Artificial**  
> Modelo Deep Learning treinado com pesos próprios para triagem, classificação e catalogação automática de documentos de seleções *Stricto Sensu* (Mestrado e Doutorado) das instituições públicas do Piauí (UFPI, UESPI, IFPI e UFDPAR).

---

## 1. Problema de Nicho e Usuário Real

* **Usuário-Alvo:** Pró-Reitorias de Pesquisa e Pós-Graduação (PRPG/PROP), secretarias acadêmicas de programas *Stricto Sensu*, portais integradores de editais e candidatos.
* **Dor Real:** A triagem manual de centenas de arquivos PDF publicados diariamente em formatos fragmentados consome horas de trabalho operacional e gera atrasos em publicações de resultados, recursos e homologações.
* **Entrada do Modelo:** Cabeçalho e texto da primeira página do PDF (mínimo de 10 caracteres).
* **Saída da Rede:** Predição probabilística entre as **6 classes oficiais**:
  1. `edital` (Abertura de seleção / normas gerais)
  2. `homologacao` (Inscrições aceitas / indeferidas)
  3. `recurso` (Formulários e decisões de recursos)
  4. `resultado` (Notas, classificação preliminar e final)
  5. `aviso` (Convocação para matrículas, bancas e comunicados)
  6. `retificacao` (Alterações de cronograma ou termos do edital)

---

## 2. Arquitetura Neural Do Zero (PyTorch)

A rede neural foi implementada em [`src/models/network.py`](src/models/network.py) com **100% dos pesos inicializados aleatoriamente**, sem depender de modelos pré-treinados ou APIs externas proprietárias:

```text
[Texto do Edital] 
       │
       ▼  (Tokenização & Vocabulário próprio de treino: 1.249 palavras)
[Tensor de IDs: Shape (Batch, 128)]
       │
       ▼  Camada de Embedding (1.249 x 128 = 159.872 parâmetros treináveis)
[Tensor Embebido: Shape (Batch, 128, 128)]
       │
       ▼  Global Average Pooling (Deep Averaging Network sobre a sequência)
[Vetor Compacto do Documento: Shape (Batch, 128)]
       │
       ▼  Camada Linear Oculta (128 -> 64) + Ativação ReLU + Dropout (0.2)
[Representação Latente: Shape (Batch, 64)]
       │
       ▼  Camada Linear de Saída (64 -> 6 classes)
[Logits: Shape (Batch, 6)] ──► Softmax ──► [Probabilidades das 6 Classes]
```

* **Total de Parâmetros Treináveis:** `164.550` parâmetros.
* **Hardware:** Otimizado para **CPU-first**, garantindo execução rápida em qualquer notebook convencional (latência média de inferência: **1 a 4 ms**).

---

## 3. Pipeline de Dados e MLOps

* **Dataset Curado:** 68 documentos reais de processos seletivos públicos do Piauí organizados em `data/raw/` e versionados com DVC (`data/raw.dvc`).
* **Divisão Estratificada (Sem Data Leakage):**
  * `Treino (70% - 46 docs)`: Utilizado exclusivamente para ajuste dos pesos e criação do vocabulário.
  * `Validação (15% - 11 docs)`: Utilizado para seleção de hiperparâmetros e *Early Stopping*.
  * `Teste Reservado (15% - 11 docs)`: Conjunto intocado até a avaliação final cega.
* **Rastreabilidade com MLflow:** Hiperparâmetros (`params.yaml`), métricas de época, curvas de perda e artefatos registrados no banco `mlflow.db`.

---

## 4. Resultados e Métricas Oficiais

| Abordagem | Acurácia (Validação) | Acurácia (Teste Cego) | Latência Média | Parâmetros Treináveis |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline Clássico (TF-IDF + LogReg)** | 70.00% | 54.55% | 0.8 ms | ~1.500 coefs |
| **Deep Learning Próprio (PyTorch - best.pt)** | **72.73%** | **54.55%** | **1.3 ms** | **164.550** |

* As curvas de aprendizado estão salvas em [`reports/figures/training_curves.png`](reports/figures/training_curves.png).
* A matriz de confusão do conjunto de teste está salva em [`reports/figures/confusion_matrix.png`](reports/figures/confusion_matrix.png).
* O relatório qualitativo de erros está registrado em [`reports/metrics/error_analysis.json`](reports/metrics/error_analysis.json).

---

## 5. API REST com FastAPI e Contrato Pydantic

A API em [`src/api/main.py`](src/api/main.py) disponibiliza os seguintes contratos:

* `GET /health`: Diagnóstico de saúde da aplicação, modelo carregado e classes suportadas.
* `POST /predict`: Classificação em tempo real via JSON payload (`{"texto": "..."}`).
* `POST /predict/pdf`: Upload multipart de arquivos PDF, extração de texto em memória e predição instantânea.
* `GET /`: Interface web interativa em HTML5/CSS moderno.
* `GET /docs`: Documentação interativa OpenAPI / Swagger UI.

---

## 6. Empacotamento Docker

O sistema está 100% conteinerizado para garantir execução idêntica em qualquer ambiente:

```bash
# Construir a imagem Docker
docker build -t academicia:1.0 .

# Rodar o container
docker run --rm -p 8000:8000 academicia:1.0

# Ou via Docker Compose
docker compose up -d
```

---

## 7. Como Reproduzir do Zero em uma Pasta Limpa

Para reproduzir e auditar todo o projeto em uma máquina limpa, siga os passos abaixo:

### Passo A: Clonar o Repositório
```bash
git clone https://github.com/ismael0800/AcademicIA_PIT_2026.git auditoria-academicia
cd auditoria-academicia
```

### Passo B: Criar o Ambiente Virtual e Instalar Dependências
```powershell
# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Passo C: Executar a Auditoria Automatizada em 1 Comando
O repositório inclui scripts de auditoria automatizada que validam ambiente, testes, checkpoint e artefatos:
```powershell
# Windows
.\reproduce.ps1
```
```bash
# Linux / macOS
chmod +x reproduce.sh
./reproduce.sh
```

### Passo D: Iniciar a Aplicação Web
```powershell
& .\.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000 --reload
```
Acesse no navegador:
* **Interface Web do Usuário:** `http://localhost:8000`
* **Swagger da API:** `http://localhost:8000/docs`

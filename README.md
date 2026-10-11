# AcademicIA — Classificador de Documentos de Pós-Graduação (PI)

Projeto Integrador · Inteligência Artificial de Nicho do Zero

## 1. Problema e Usuário
* **Usuário:** Secretarias acadêmicas de pós-graduação, portais integradores de seleção e candidatos.
* **Problema:** Triagem e catalogação automatizada de milhares de documentos em PDF de seleções públicas de mestrado e doutorado das instituições do Piauí (UFPI, UESPI, IFPI, FAPEPI).
* **Entrada:** Texto inicial (primeira página / cabeçalho) do documento em PDF.
* **Saída:** Uma das 5 classes: `Edital`, `Homologação`, `Recurso`, `Resultado Final` ou `Aviso/Retificação`.

## 2. Dados
* **Fonte:** Portais públicos oficiais das IES do Piauí (SIGAA UFPI, PROP UESPI).
* **Camadas:** `data/raw` (PDFs originais organizados por classe), `data/processed` (textos extraídos) e `data/curated` (dataset final splitado).
* **Divisão dos Dados:** 70% Treino, 15% Validação e 15% Teste reservado (congelado).

## 3. Baseline
* **Método:** Modelo clássico TF-IDF + Naive Bayes / Regressão Logística (scikit-learn).
* **Objetivo:** Estabelecer a referência mínima de acurácia e F1-Score antes do treino da rede neural.

## 4. Arquitetura Neural (Do Zero em PyTorch)
* **Modelo:** `TextClassifier` (Embeddings próprios + Average Pooling + Camadas Densas Lineares + ReLU).
* **Pesos:** 100% inicializados aleatoriamente e treinados do zero (sem uso de modelos pré-treinados externos).
* **Saída:** 5 logits com `CrossEntropyLoss`.

## 5. Treinamento
* **Otimizador:** AdamW.
* **Hardware:** CPU-first (executável em qualquer computador pessoal).
* **Rastreabilidade:** Hiperparâmetros centralizados em `params.yaml`.

## 6. MLOps e Reprodutibilidade
* **Ambiente isolado:** Virtualenv (`.venv`).
* **Versionamento de Código:** Git.
* **Rastreamento de Experimentos:** MLflow.
* **Pipeline e Dados:** DVC.
* **Testes Automatizados:** PyTest.

## 7. Como Executar
```powershell
# 1. Ativar o ambiente virtual
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependências
pip install -r requirements.txt
```

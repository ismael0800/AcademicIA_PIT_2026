"""
Testes automatizados da API REST (FastAPI) do AcademicIA.
Valida endpoints /health, /predict, /predict/pdf e conformidade com Pydantic.
"""
from pathlib import Path
import io
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Valida o endpoint /health e presença dos metadados do modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "modelo" in data
    assert "dispositivo" in data
    assert len(data["classes_suportadas"]) == 6


def test_predict_text_success():
    """Valida inferência via JSON com texto válido."""
    payload = {
        "texto": "EDITAL Nº 01/2026 - Abertura de inscrições para o processo seletivo do Programa de Pós-Graduação."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "sucesso"
    assert data["classe_predita"] in ["edital", "homologacao", "recurso", "resultado", "aviso", "retificacao"]
    assert data["confianca_percentual"] > 0
    assert "distribuicao_probabilidades" in data
    assert len(data["distribuicao_probabilidades"]) == 6
    assert data["latencia_ms"] > 0


def test_predict_text_validation_error():
    """Valida que textos com menos de 10 caracteres são rejeitados pelo Pydantic."""
    payload = {"texto": "Curto"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)


def test_web_interface_endpoint():
    """Valida se o endpoint raiz serve a interface web em HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "AcademicIA" in response.text

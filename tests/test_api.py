import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_api_query():
    payload = {"query": "Transformer self-attention models"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "strategy" in data

def test_api_graph():
    response = client.get("/graph")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data

def test_api_evaluate():
    payload = {
        "dataset": [
            {"query": "self-attention mechanism", "reference": "Self-attention mechanism maps sequence elements."}
        ]
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "averages" in data
    assert "records" in data

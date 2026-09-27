import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient

from main import app

cliente = TestClient(app)


def test_predict_sucesso():
    resposta = cliente.post("/predict", json={"atributos": ["cavalos", "torque máximo"]})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert "modeloVersao" in corpo
    assert len(corpo["resultados"]) == 2
    for item in corpo["resultados"]:
        assert set(item.keys()) == {"entrada", "atributo", "score"}
        assert 0.0 <= item["score"] <= 1.0


def test_predict_lista_vazia_e_invalida():
    resposta = cliente.post("/predict", json={"atributos": []})
    assert resposta.status_code == 422


def test_predict_atributo_muito_longo_e_invalido():
    texto_longo = "a" * 61
    resposta = cliente.post("/predict", json={"atributos": [texto_longo]})
    assert resposta.status_code == 422


def test_predict_atributo_com_caracteres_de_controle_e_sanitizado():
    resposta = cliente.post("/predict", json={"atributos": ["cavalos\x00\x1f"]})
    assert resposta.status_code == 200
    assert resposta.json()["resultados"][0]["entrada"] == "cavalos"


def test_predict_limite_de_50_atributos_ok():
    atributos = ["cavalos"] * 50
    resposta = cliente.post("/predict", json={"atributos": atributos})
    assert resposta.status_code == 200
    assert len(resposta.json()["resultados"]) == 50


def test_predict_acima_de_50_atributos_e_invalido():
    atributos = ["cavalos"] * 51
    resposta = cliente.post("/predict", json={"atributos": atributos})
    assert resposta.status_code == 422


def test_health():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "ok"


def test_metrics():
    resposta = cliente.get("/metrics")
    assert resposta.status_code == 200
    assert "specradar_ml_confidence" in resposta.text

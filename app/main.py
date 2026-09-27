import json
import re
import sys
import time
import uuid
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO / "src"))

from preprocessamento import VetorizadorCombinado  # noqa: E402  (necessario para o unpickle do joblib)

CAMINHO_MODELO = RAIZ_PROJETO / "models" / "modelo.joblib"
CAMINHO_METADATA = RAIZ_PROJETO / "models" / "metadata.json"

PADRAO_CARACTERES_CONTROLE = re.compile(r"[\x00-\x1f\x7f]")
NAO_DISPONIVEL = "Não disponível"

app = FastAPI(title="SpecRadar ML", version="1.0.0")

modelo = joblib.load(CAMINHO_MODELO)
with open(CAMINHO_METADATA, "r", encoding="utf-8") as arquivo:
    metadata = json.load(arquivo)

VERSAO_MODELO = metadata["versao"]
LIMIAR_CONFIANCA = metadata["limiar_confianca"]

confianca_hist = Histogram(
    "specradar_ml_confidence", "Score de confianca das predicoes do classificador"
)
nao_disponivel_ratio = Gauge(
    "specradar_ml_not_available_ratio",
    "Proporcao acumulada de resultados classificados como Nao disponivel",
)
inferencia_hist = Histogram(
    "specradar_ml_inference_seconds", "Tempo de inferencia por requisicao, em segundos"
)

_contador_total = 0
_contador_nao_disponivel = 0


def sanitizar_atributo(valor: str) -> str:
    sem_controle = PADRAO_CARACTERES_CONTROLE.sub("", valor)
    return sem_controle.strip()


class ItemAtributo(BaseModel):
    valor: str = Field(min_length=1, max_length=60)

    @field_validator("valor")
    @classmethod
    def validar_e_sanitizar(cls, valor: str) -> str:
        valor_limpo = sanitizar_atributo(valor)
        if not valor_limpo:
            raise ValueError("atributo vazio apos sanitizacao")
        if len(valor_limpo) > 60:
            raise ValueError("atributo excede 60 caracteres")
        return valor_limpo


class PredictRequest(BaseModel):
    atributos: list[str] = Field(min_length=1, max_length=50)

    @field_validator("atributos")
    @classmethod
    def validar_lista(cls, atributos: list[str]) -> list[str]:
        return [ItemAtributo(valor=item).valor for item in atributos]


class ResultadoPredicao(BaseModel):
    entrada: str
    atributo: str
    score: float


class PredictResponse(BaseModel):
    resultados: list[ResultadoPredicao]
    modeloVersao: str


def registrar_log_baixa_confianca(trace_id: str, entrada: str, atributo: str, score: float):
    registro = {
        "evento": "ML_LOW_CONFIDENCE",
        "traceId": trace_id,
        "entrada": entrada,
        "atributoPrevisto": atributo,
        "score": round(score, 4),
    }
    print(json.dumps(registro, ensure_ascii=False))


@app.post("/predict", response_model=PredictResponse)
def predizer(payload: PredictRequest, request: Request):
    global _contador_total, _contador_nao_disponivel

    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))

    inicio = time.perf_counter()
    probabilidades = modelo.predict_proba(payload.atributos)
    duracao = time.perf_counter() - inicio
    inferencia_hist.observe(duracao)

    classes = modelo.classes_
    resultados = []
    for texto_entrada, linha_probabilidades in zip(payload.atributos, probabilidades):
        indice_max = linha_probabilidades.argmax()
        score = float(linha_probabilidades[indice_max])
        atributo_previsto = classes[indice_max]

        confianca_hist.observe(score)
        _contador_total += 1

        if score < LIMIAR_CONFIANCA:
            atributo_final = NAO_DISPONIVEL
            _contador_nao_disponivel += 1
            registrar_log_baixa_confianca(trace_id, texto_entrada, atributo_previsto, score)
        else:
            atributo_final = atributo_previsto

        resultados.append(
            ResultadoPredicao(entrada=texto_entrada, atributo=atributo_final, score=round(score, 4))
        )

    nao_disponivel_ratio.set(_contador_nao_disponivel / _contador_total)

    return PredictResponse(resultados=resultados, modeloVersao=VERSAO_MODELO)


@app.get("/health")
def health():
    return {"status": "ok", "modeloVersao": VERSAO_MODELO}


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(ValueError)
def tratar_erro_validacao(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

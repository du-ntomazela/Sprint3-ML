# Participantes
- Eduardo Tomazela do Nascimento rm556807
- Léo Masago rm557768
- Luiz Henrique Silva rm555735
- 
# SpecRadar ML

Desafio Ford 01 – Inteligência Competitiva Automotiva | FIAP Sprint 3 – IA & Machine Learning

## Problema

O usuário do SpecRadar digita livremente o atributo que quer pesquisar sobre um veículo ("cavalos", "força do motor", "hp", "torque max", "tamanho da caçamba"...). Este serviço classifica cada texto curto digitado em um **atributo canônico** do catálogo (`potencia`, `torque`, `cilindrada`, `comprimento`, `capacidade_cacamba` etc.) ou em `fora_do_catalogo`, quando o termo não pertence ao domínio de especificações do veículo. É um problema de **classificação multiclasse de texto curto**: a cada predição é associado um score de confiança que decide se o campo é exibido normalmente ou como "Não disponível".

## Dataset

Sintético, gerado por [`src/gerar_dataset.py`](src/gerar_dataset.py): ~40 atributos canônicos, cada um expandido em 60–120 variações (sinônimos PT/EN, abreviações, erros de digitação, unidades de medida, variações de caixa/acentuação) a partir de frases-semente, mais a classe `fora_do_catalogo`. O gerador injeta de propósito nulos, duplicatas, rótulos inconsistentes e outliers, tratados depois em [`src/preprocessamento.py`](src/preprocessamento.py) (normalização + limpeza). Features: TF-IDF de caracteres (2–5) + TF-IDF de palavras + variáveis extras (tamanho do texto, presença de dígito/unidade). Divisão 80/20 estratificada, `random_state=42`.

## Comparação dos modelos

Resultado da última execução completa do notebook (conjunto de teste, 654 exemplos, 41 classes):

| Modelo | Acurácia | Precisão macro | Recall macro | F1 macro | Inferência (ms/amostra) |
|---|---|---|---|---|---|
| **Regressão Logística** | 0,9817 | 0,9824 | 0,9807 | 0,9808 | 0,070 |
| Random Forest | 0,9801 | 0,9810 | 0,9804 | 0,9799 | 0,166 |
| MLP (rede neural) | 0,9786 | 0,9790 | 0,9794 | 0,9783 | 0,074 |
| Linear SVM (calibrado) | 0,9786 | 0,9802 | 0,9776 | 0,9779 | 0,091 |
| KNN | 0,5948 | 0,6317 | 0,5928 | 0,5930 | 0,136 |
| Naive Bayes Multinomial | 0,3303 | 0,7495 | 0,2675 | 0,3331 | 0,069 |

KNN e Naive Bayes sofrem com a alta dimensionalidade esparsa do TF-IDF combinado a muitas classes com poucos exemplos cada — respectivamente a maldição da dimensionalidade na distância euclidiana e o desequilíbrio de escala entre features de contagem e features extras violando a suposição de independência do Naive Bayes.

## Modelo escolhido

**Regressão Logística**, ajustada com `GridSearchCV` (5-fold, `C=5.0`), atingiu **F1 macro de 0,9859** na validação cruzada e **0,9779** no conjunto de teste após o ajuste — o melhor equilíbrio entre acurácia e tempo de inferência (~0,068 ms/amostra) entre os seis modelos avaliados. Empates estreitos com Random Forest, MLP e SVM calibrado são esperados nesse regime (poucas centenas de exemplos por classe, features lineares bem separáveis); ver [`notebooks/specradar_ml.ipynb`](notebooks/specradar_ml.ipynb) para a análise completa, incluindo matriz de confusão, análise de erros e a curva de cobertura × acurácia usada para escolher o limiar de confiança (0,6).

A validação com a lista de atributos da Ford Ranger Raptor (`dados/ranger_raptor_atributos.txt`) reconheceu corretamente 16 de 17 termos com score acima de 0,97; o único caso abaixo do limiar ("vau máximo") foi corretamente sinalizado como "Não disponível" em vez de arriscar uma resposta errada.

## Como treinar

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python src\gerar_dataset.py
.venv\Scripts\python src\treinar.py
```

Isso gera `dados/dataset_bruto.csv`, `dados/dataset_limpo.csv`, `models/modelo.joblib` e `models/metadata.json` (versão do modelo e métricas de teste).

## Como rodar o notebook

```bash
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace notebooks\specradar_ml.ipynb
```

ou abra `notebooks/specradar_ml.ipynb` normalmente no Jupyter/VS Code e execute célula a célula.

## Como subir o serviço

```bash
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000
```

ou via Docker:

```bash
docker build -t specradar-ml .
docker run -p 8000:8000 specradar-ml
```

### Contrato da API

```
POST /predict
{"atributos": ["cavalos", "torque máximo"]}

200 OK
{
  "resultados": [
    {"entrada": "cavalos", "atributo": "potencia", "score": 0.93},
    {"entrada": "torque máximo", "atributo": "torque", "score": 0.91}
  ],
  "modeloVersao": "1.0.0"
}
```

- `atributos`: lista de 1 a 50 strings, cada uma com até 60 caracteres (validado e sanitizado — caracteres de controle são removidos).
- `GET /health`: status do serviço e versão do modelo carregado.
- `GET /metrics`: métricas Prometheus (`specradar_ml_confidence`, `specradar_ml_not_available_ratio`, `specradar_ml_inference_seconds`).
- Quando o score de uma predição fica abaixo de 0,6, o serviço registra um log JSON com evento `ML_LOW_CONFIDENCE` (propagando o header `X-Trace-Id`) e retorna `"atributo": "Não disponível"` para aquele item.

## Testes

```bash
.venv\Scripts\pytest tests\ -v
```

## Estrutura

```
specradar-ml/
├── notebooks/specradar_ml.ipynb   # notebook completo, executado, com saídas salvas
├── src/
│   ├── gerar_dataset.py           # geração do dataset sintético
│   ├── preprocessamento.py        # normalização, limpeza, features
│   └── treinar.py                 # treino, comparação, GridSearchCV, salvamento
├── app/main.py                    # serviço FastAPI (contrato consumido pela API Java)
├── dados/                         # dataset bruto/limpo + lista de atributos da Ranger Raptor
├── models/                        # modelo.joblib + metadata.json (gerados pelo treino)
├── tests/test_main.py             # testes do endpoint /predict
├── Dockerfile
├── requirements.txt               # dependências de produção (fixas)
└── requirements-dev.txt           # + pytest, jupyter (dev/notebook)
```

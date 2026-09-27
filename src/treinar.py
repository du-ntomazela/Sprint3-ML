import json
import time
import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from preprocessamento import VetorizadorCombinado, limpar_dataset, normalizar_lista

RANDOM_STATE = 42
VERSAO_MODELO = "1.0.0"
LIMIAR_CONFIANCA = 0.6

CAMINHO_BRUTO = "dados/dataset_bruto.csv"
CAMINHO_LIMPO = "dados/dataset_limpo.csv"
CAMINHO_MODELO = "models/modelo.joblib"
CAMINHO_METADATA = "models/metadata.json"

DEFINICOES_MODELOS = {
    "Naive Bayes Multinomial": MultinomialNB(),
    "Regressao Logistica": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Linear SVM (calibrado)": CalibratedClassifierCV(
        LinearSVC(random_state=RANDOM_STATE, max_iter=3000), cv=3
    ),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "MLP (rede neural)": MLPClassifier(
        hidden_layer_sizes=(128,), max_iter=400, random_state=RANDOM_STATE
    ),
}

GRADES_HIPERPARAMETROS = {
    "Naive Bayes Multinomial": {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
    "Regressao Logistica": {"clf__C": [0.1, 1.0, 5.0, 10.0]},
    "Linear SVM (calibrado)": {"clf__estimator__C": [0.1, 1.0, 5.0]},
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 40],
    },
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9]},
    "MLP (rede neural)": {
        "clf__alpha": [0.0001, 0.001],
        "clf__hidden_layer_sizes": [(64,), (128,)],
    },
}


def montar_pipeline(modelo):
    return Pipeline(
        [
            ("normalizar", FunctionTransformer(normalizar_lista)),
            ("vetor", VetorizadorCombinado()),
            ("clf", modelo),
        ]
    )


def carregar_dados_limpos():
    df_bruto = pd.read_csv(CAMINHO_BRUTO)
    df_limpo = limpar_dataset(df_bruto)
    Path(CAMINHO_LIMPO).parent.mkdir(parents=True, exist_ok=True)
    df_limpo.to_csv(CAMINHO_LIMPO, index=False)
    return df_limpo


def dividir_treino_teste(df_limpo):
    return train_test_split(
        df_limpo["texto"].tolist(),
        df_limpo["atributo"].tolist(),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df_limpo["atributo"],
    )


def avaliar_modelo(pipeline, X_test, y_test):
    inicio = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    duracao = time.perf_counter() - inicio
    tempo_medio_ms = (duracao / len(X_test)) * 1000

    return {
        "acuracia": accuracy_score(y_test, y_pred),
        "precisao_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "tempo_inferencia_ms_por_amostra": tempo_medio_ms,
    }, y_pred


def treinar_todos_modelos(X_train, y_train, X_test, y_test):
    pipelines_treinados = {}
    linhas_metricas = []

    for nome, modelo in DEFINICOES_MODELOS.items():
        pipeline = montar_pipeline(modelo)
        pipeline.fit(X_train, y_train)
        metricas, _ = avaliar_modelo(pipeline, X_test, y_test)
        metricas["modelo"] = nome
        linhas_metricas.append(metricas)
        pipelines_treinados[nome] = pipeline

    df_metricas = pd.DataFrame(linhas_metricas).set_index("modelo")
    df_metricas = df_metricas.sort_values("f1_macro", ascending=False)
    return df_metricas, pipelines_treinados


def ajustar_hiperparametros(nome_melhor, X_train, y_train):
    modelo_base = DEFINICOES_MODELOS[nome_melhor]
    pipeline_base = montar_pipeline(modelo_base)
    grade = GRADES_HIPERPARAMETROS[nome_melhor]

    busca = GridSearchCV(
        pipeline_base,
        param_grid=grade,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    busca.fit(X_train, y_train)
    return busca


def calcular_curva_cobertura(pipeline, X_test, y_test, limiares=None):
    if limiares is None:
        limiares = np.arange(0.1, 1.0, 0.05)

    probabilidades = pipeline.predict_proba(X_test)
    classes = pipeline.classes_
    indices_max = probabilidades.argmax(axis=1)
    scores_max = probabilidades.max(axis=1)
    predicoes = classes[indices_max]

    linhas = []
    y_test_array = np.array(y_test)
    for limiar in limiares:
        mascara_aceitas = scores_max >= limiar
        cobertura = mascara_aceitas.mean()
        if mascara_aceitas.sum() > 0:
            acuracia_aceitas = (predicoes[mascara_aceitas] == y_test_array[mascara_aceitas]).mean()
        else:
            acuracia_aceitas = np.nan
        linhas.append({"limiar": round(float(limiar), 2), "cobertura": cobertura, "acuracia": acuracia_aceitas})

    return pd.DataFrame(linhas)


def main():
    df_limpo = carregar_dados_limpos()
    X_train, X_test, y_train, y_test = dividir_treino_teste(df_limpo)

    df_metricas, pipelines_treinados = treinar_todos_modelos(X_train, y_train, X_test, y_test)
    nome_melhor = df_metricas.index[0]

    busca = ajustar_hiperparametros(nome_melhor, X_train, y_train)
    melhor_pipeline = busca.best_estimator_
    metricas_finais, y_pred_final = avaliar_modelo(melhor_pipeline, X_test, y_test)

    matriz_confusao = confusion_matrix(y_test, y_pred_final, labels=melhor_pipeline.classes_)

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor_pipeline, CAMINHO_MODELO)

    metadata = {
        "versao": VERSAO_MODELO,
        "modelo_escolhido": nome_melhor,
        "melhores_hiperparametros": busca.best_params_,
        "limiar_confianca": LIMIAR_CONFIANCA,
        "metricas_teste": metricas_finais,
        "classes": sorted(melhor_pipeline.classes_.tolist()),
        "n_amostras_treino": len(X_train),
        "n_amostras_teste": len(X_test),
    }
    with open(CAMINHO_METADATA, "w", encoding="utf-8") as arquivo:
        json.dump(metadata, arquivo, ensure_ascii=False, indent=2)

    print(df_metricas)
    print("\nMelhor modelo:", nome_melhor)
    print("Metricas finais (pos GridSearchCV):", metricas_finais)

    return {
        "df_metricas": df_metricas,
        "pipelines_treinados": pipelines_treinados,
        "melhor_pipeline": melhor_pipeline,
        "nome_melhor": nome_melhor,
        "busca": busca,
        "metricas_finais": metricas_finais,
        "matriz_confusao": matriz_confusao,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_final": y_pred_final,
        "df_limpo": df_limpo,
    }


if __name__ == "__main__":
    main()

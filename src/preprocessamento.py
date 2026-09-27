import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

PADRAO_UNIDADE = re.compile(
    r"\b(cv|hp|ps|nm|kgfm|km/h|km/l|kg|mm|cm3|l|litros|graus|polegadas|r\$|reais|s|anos)\b"
)
PADRAO_DIGITO = re.compile(r"\d")
PADRAO_PONTUACAO = re.compile(r"[^\w\s]")
PADRAO_ESPACOS = re.compile(r"\s+")


def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = PADRAO_PONTUACAO.sub(" ", texto)
    texto = PADRAO_ESPACOS.sub(" ", texto).strip()
    return texto


def normalizar_lista(textos):
    return [normalizar_texto(t) for t in textos]


def limpar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["texto"] = df["texto"].apply(normalizar_texto)

    df = df[df["texto"].str.len() > 0].reset_index(drop=True)

    contagem_por_texto = df.groupby("texto")["atributo"].transform(
        lambda serie: serie.value_counts().idxmax()
    )
    df["atributo"] = contagem_por_texto

    df = df.drop_duplicates(subset=["texto"]).reset_index(drop=True)
    return df


class FeaturesExtras(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        tamanhos = np.array([len(t) for t in X], dtype=float).reshape(-1, 1)
        contem_digito = np.array(
            [1.0 if PADRAO_DIGITO.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        contem_unidade = np.array(
            [1.0 if PADRAO_UNIDADE.search(t) else 0.0 for t in X]
        ).reshape(-1, 1)
        n_palavras = np.array(
            [len(t.split()) for t in X], dtype=float
        ).reshape(-1, 1)
        return np.hstack([tamanhos, contem_digito, contem_unidade, n_palavras])

    def get_feature_names_out(self, input_features=None):
        return np.array(["tamanho_texto", "contem_digito", "contem_unidade", "n_palavras"])


class VetorizadorCombinado(BaseEstimator, TransformerMixin):
    """Combina TF-IDF de caracteres (2-5) + TF-IDF de palavras + features extras."""

    def __init__(self):
        self.tfidf_char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=4000
        )
        self.tfidf_palavra = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=3000
        )
        self.features_extras = FeaturesExtras()

    def fit(self, X, y=None):
        self.tfidf_char.fit(X)
        self.tfidf_palavra.fit(X)
        self.features_extras.fit(X)
        return self

    def transform(self, X):
        mat_char = self.tfidf_char.transform(X)
        mat_palavra = self.tfidf_palavra.transform(X)
        mat_extras = self.features_extras.transform(X)
        return hstack([mat_char, mat_palavra, mat_extras]).tocsr()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

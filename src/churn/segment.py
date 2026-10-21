"""Segmentación de clientes por comportamiento (K-Means sobre recencia, frecuencia, gasto y antigüedad)."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn import config

# Variables tipo RFM (+ antigüedad). No incluye `Churn`: la segmentación no usa la etiqueta.
SEGMENT_FEATURES: tuple[str, ...] = ("Tenure", "DaySinceLastOrder", "OrderCount", "CashbackAmount")

FEATURE_LABELS: dict[str, str] = {
    "Tenure": "antigüedad",
    "DaySinceLastOrder": "días sin comprar",
    "OrderCount": "órdenes",
    "CashbackAmount": "cashback",
}

DEFAULT_K_RANGE = range(3, 7)
MAX_SILHOUETTE_SAMPLE = 5000
DEVIATION_THRESHOLD = 0.15


def _require_features(df: pd.DataFrame) -> None:
    missing = [c for c in SEGMENT_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para segmentar: {missing}")


def _preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def fit_segmenter(
    df: pd.DataFrame,
    k_range: Iterable[int] = DEFAULT_K_RANGE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[Pipeline, dict[str, object]]:
    """Ajusta K-Means eligiendo k con mayor coeficiente de silueta.

    Devuelve el pipeline ajustado (imputación + escalado + K-Means) y un resumen
    con el `k` elegido y la silueta de cada candidato.
    """
    _require_features(df)
    features = df[list(SEGMENT_FEATURES)]
    scaled = _preprocessor().fit_transform(features)
    sample_size = min(len(df), MAX_SILHOUETTE_SAMPLE)

    silhouette_by_k: dict[int, float] = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(scaled)
        silhouette_by_k[k] = float(
            silhouette_score(scaled, labels, sample_size=sample_size, random_state=random_state)
        )

    best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    segmenter = Pipeline(
        [
            ("prep", _preprocessor()),
            ("kmeans", KMeans(n_clusters=best_k, n_init=10, random_state=random_state)),
        ]
    ).fit(features)

    info = {
        "k": best_k,
        "silhouette_by_k": silhouette_by_k,
        "features": list(SEGMENT_FEATURES),
    }
    return segmenter, info


def assign_segments(segmenter: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Asigna cada cliente a su segmento (0..k-1)."""
    _require_features(df)
    return segmenter.predict(df[list(SEGMENT_FEATURES)])


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Tabla de perfil por segmento: tamaño, promedios de las variables y tasa de churn real."""
    _require_features(df)
    frame = df[list(SEGMENT_FEATURES)].assign(segmento=np.asarray(labels))
    grouped = frame.groupby("segmento")
    profile = grouped.mean().round(2)
    profile.insert(0, "clientes", grouped.size())
    if config.TARGET in df.columns:
        churn_rate = df[config.TARGET].groupby(np.asarray(labels)).mean().round(4)
        profile.insert(1, "tasa_churn", churn_rate.rename_axis("segmento"))
    return profile


def describe_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Descripción breve de cada segmento según cuánto se aleja del promedio general."""
    weights = profile["clientes"]
    overall = {f: float(np.average(profile[f], weights=weights)) for f in SEGMENT_FEATURES}
    descriptions: dict[int, str] = {}
    for segment, row in profile.iterrows():
        parts = []
        for feature in SEGMENT_FEATURES:
            if overall[feature] == 0:
                continue
            deviation = row[feature] / overall[feature] - 1
            if abs(deviation) >= DEVIATION_THRESHOLD:
                direction = "mayor" if deviation > 0 else "menor"
                parts.append(f"{FEATURE_LABELS[feature]} {direction} ({deviation:+.0%})")
        descriptions[int(segment)] = ", ".join(parts) or "perfil cercano al promedio"
    return descriptions

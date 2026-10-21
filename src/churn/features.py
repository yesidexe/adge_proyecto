"""Preprocesamiento: imputación, escalado y codificación one-hot."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Construye el preprocesador (sin ajustar).

    - Numéricas: imputación por mediana y, si `scale`, estandarización.
    - Categóricas: one-hot (las categorías nuevas en predicción se ignoran).

    Al ir dentro de un `Pipeline`, la mediana y el escalado se calculan solo con
    los datos de entrenamiento de cada partición (sin *data leakage*).
    """
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(config.NUMERIC_COLUMNS)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(config.CATEGORICAL_COLUMNS),
            ),
        ],
        sparse_threshold=0.0,
    )

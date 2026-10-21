"""Definición de los dos modelos: Regresión Logística (base) y XGBoost (avanzado)."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.features import build_preprocessor

LOGISTIC = "Regresion Logistica"
XGBOOST = "XGBoost"
MODEL_NAMES: tuple[str, ...] = (LOGISTIC, XGBOOST)

LOGISTIC_PARAMS: dict[str, object] = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": config.RANDOM_STATE,
}

XGBOOST_PARAMS: dict[str, object] = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": config.RANDOM_STATE,
    "eval_metric": "logloss",
}


def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """Crea los pipelines (sin entrenar) con manejo del desbalance de clases.

    - Regresión Logística: requiere variables escaladas y `class_weight='balanced'`.
    - XGBoost: no necesita escalado; `scale_pos_weight` = retenidos / desertores.
    """
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("y_train debe contener clientes de ambas clases (0 y 1).")

    return {
        LOGISTIC: Pipeline(
            [
                ("preprocess", build_preprocessor(scale=True)),
                ("model", LogisticRegression(**LOGISTIC_PARAMS)),
            ]
        ),
        XGBOOST: Pipeline(
            [
                ("preprocess", build_preprocessor(scale=False)),
                ("model", XGBClassifier(scale_pos_weight=negatives / positives, **XGBOOST_PARAMS)),
            ]
        ),
    }

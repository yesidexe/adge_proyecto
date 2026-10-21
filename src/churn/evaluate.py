"""Métricas de clasificación y criterio de selección del modelo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SELECTION_RULE = (
    "mayor Recall medio en validación cruzada (desempate por ROC-AUC medio); "
    "el conjunto de prueba se reserva para el reporte final"
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> dict[str, float]:
    """Métricas de negocio para churn + conteos de la matriz de confusión."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def select_best_model(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    recall_key: str = "recall",
    auc_key: str = "roc_auc",
) -> str:
    """Elige el modelo con mayor Recall (métrica principal); desempata por ROC-AUC.

    Se espera recibir métricas de validación cruzada (`recall_mean`, `roc_auc_mean`)
    para no seleccionar el modelo mirando el conjunto de prueba.
    """
    if not metrics_by_model:
        raise ValueError("No hay métricas para seleccionar un modelo.")
    return max(
        metrics_by_model,
        key=lambda name: (metrics_by_model[name][recall_key], metrics_by_model[name][auc_key]),
    )

"""Entrenamiento, evaluación y exportación de artefactos.

Uso:
    python -m churn.train

Genera `models/` (modelo + segmentador), `reports/` (métricas, gráficos, reporte)
y `data/processed/` (predicciones de prueba y segmentos por cliente).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from churn import config
from churn.data import (
    load_raw,
    split_features_target,
    standardize_categories,
    train_test_split_stratified,
)
from churn.evaluate import SELECTION_RULE, compute_metrics, select_best_model
from churn.logging_setup import setup_logging
from churn.modeling import LOGISTIC, LOGISTIC_PARAMS, XGBOOST, XGBOOST_PARAMS, build_models
from churn.reporting import (
    TOP_N_FEATURES,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_roc_curves,
    plot_segments,
    write_training_report,
)
from churn.segment import assign_segments, describe_segments, fit_segmenter, profile_segments

logger = logging.getLogger(__name__)

CV_SCORING = {"recall": "recall", "precision": "precision", "f1": "f1", "roc_auc": "roc_auc"}


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    test_metrics: dict[str, dict[str, float]]
    cv_metrics: dict[str, dict[str, float]]
    segments_k: int
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class Evaluation:
    """Modelos entrenados con sus métricas de validación cruzada y de prueba."""

    models: dict[str, Pipeline]
    cv: dict[str, dict[str, float]]
    test: dict[str, dict[str, float]]
    probs: dict[str, np.ndarray]


@dataclass(frozen=True)
class Segmentation:
    model: Pipeline
    labels: np.ndarray
    profile: pd.DataFrame
    info: dict[str, Any]
    descriptions: dict[int, str]


def _cross_validate(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=CV_SCORING, n_jobs=1)
    summary: dict[str, float] = {}
    for key in CV_SCORING:
        values = scores[f"test_{key}"]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def _fit_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int,
) -> Evaluation:
    """Validación cruzada sobre entrenamiento, ajuste final y evaluación en prueba."""
    models = build_models(y_train)
    cv: dict[str, dict[str, float]] = {}
    test: dict[str, dict[str, float]] = {}
    probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        logger.info("Entrenando %s (validación cruzada de %d particiones)...", name, cv_folds)
        cv[name] = _cross_validate(model, X_train, y_train, cv_folds)
        model.fit(X_train, y_train)
        probs[name] = model.predict_proba(X_test)[:, 1]
        y_pred = (probs[name] >= config.DECISION_THRESHOLD).astype(int)
        test[name] = compute_metrics(y_test, y_pred, probs[name])
        logger.info("%s -> Recall %.2f%% | ROC-AUC %.2f%%", name,
                    test[name]["recall"] * 100, test[name]["roc_auc"] * 100)
    return Evaluation(models, cv, test, probs)


def _feature_weights(model: Pipeline, name: str) -> pd.Series:
    """Importancia (XGBoost) o |coeficiente| (Regresión Logística) por variable codificada."""
    names = [n.split("__", 1)[1] for n in model.named_steps["preprocess"].get_feature_names_out()]
    estimator = model.named_steps["model"]
    weights = estimator.feature_importances_ if name == XGBOOST else np.abs(estimator.coef_[0])
    return pd.Series(weights, index=names)


def _segment_customers(clean: pd.DataFrame) -> Segmentation:
    """Segmentación no supervisada sobre todos los clientes (no usa la etiqueta Churn)."""
    segmenter, info = fit_segmenter(clean)
    labels = assign_segments(segmenter, clean)
    profile = profile_segments(clean, labels)
    logger.info("Segmentación: k=%d", info["k"])
    return Segmentation(segmenter, labels, profile, info, describe_segments(profile))


def _display_path(path: Path) -> str:
    """Ruta relativa a la raíz del proyecto cuando es posible (para el reporte)."""
    try:
        return path.resolve().relative_to(config.ROOT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _data_summary(raw: pd.DataFrame, source: Path) -> dict[str, Any]:
    nulls = raw.isna().sum()
    return {
        "source": _display_path(source),
        "n_rows": int(len(raw)),
        "n_cols": int(raw.shape[1]),
        "n_churn": int(raw[config.TARGET].sum()),
        "churn_rate": float(raw[config.TARGET].mean()),
        "duplicated_ids": int(raw[config.ID_COL].duplicated().sum()),
        "nulls": {c: int(n) for c, n in nulls[nulls > 0].sort_values(ascending=False).items()},
        "category_fixes": config.CATEGORY_FIXES,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _persist_data_and_models(
    selected: str,
    evaluation: Evaluation,
    segmentation: Segmentation,
    clean: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models_dir: Path,
    processed_dir: Path,
) -> None:
    """Guarda modelos (.joblib) y los datasets derivados (predicciones de prueba y segmentos)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(evaluation.models[selected], models_dir / config.MODEL_FILE)
    joblib.dump(segmentation.model, models_dir / config.SEGMENTER_FILE)

    predictions = pd.DataFrame(
        {config.ID_COL: clean.loc[X_test.index, config.ID_COL].to_numpy(), config.TARGET: y_test.to_numpy()}
    )
    for name, prob in evaluation.probs.items():
        predictions[f"prob_{name}"] = prob.round(4)
    predictions.to_csv(processed_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {config.ID_COL: clean[config.ID_COL], "segmento": segmentation.labels, config.TARGET: clean[config.TARGET]}
    ).to_csv(processed_dir / "customer_segments.csv", index=False)


def _write_reports(
    selected: str,
    evaluation: Evaluation,
    segmentation: Segmentation,
    context: dict[str, Any],
    y_test: pd.Series,
    weights: dict[str, pd.Series],
    models_dir: Path,
    reports_dir: Path,
) -> None:
    """Gráficos, `training_report.md`, `metrics.json` y `model_card.json`."""
    figures = reports_dir / "figures"
    plot_roc_curves(y_test, evaluation.probs, figures / "roc_curves.png")
    plot_confusion_matrices(evaluation.test, figures / "confusion_matrices.png")
    plot_feature_importance(weights, figures / "feature_importance.png")
    plot_segments(segmentation.profile, figures / "segments.png")
    write_training_report(reports_dir / "training_report.md", context)

    _write_json(
        reports_dir / "metrics.json",
        {
            "selected_model": selected,
            "selection_rule": SELECTION_RULE,
            "test": evaluation.test,
            "cv": evaluation.cv,
            "top_features": context["top_features"],
            "segmentation": {
                "k": segmentation.info["k"],
                "silhouette_by_k": segmentation.info["silhouette_by_k"],
                "descriptions": segmentation.descriptions,
                "profile": segmentation.profile.reset_index().to_dict(orient="records"),
            },
        },
    )
    _write_json(
        models_dir / "model_card.json",
        {
            "model": selected,
            "trained_at": context["generated_at"],
            "libraries": context["versions"],
            "random_state": config.RANDOM_STATE,
            "decision_threshold": config.DECISION_THRESHOLD,
            "risk_bands": {"medio_desde": config.RISK_MEDIUM_THRESHOLD, "alto_desde": config.RISK_HIGH_THRESHOLD},
            "features": list(config.FEATURE_COLUMNS),
            "hyperparameters": context["hyperparameters"][selected],
            "train_size": context["split"]["n_train"],
            "test_size": context["split"]["n_test"],
            "test_metrics": evaluation.test[selected],
        },
    )


def _report_context(
    selected: str,
    evaluation: Evaluation,
    segmentation: Segmentation,
    raw: pd.DataFrame,
    raw_path: Path,
    y_train: pd.Series,
    y_test: pd.Series,
    weights: dict[str, pd.Series],
    cv_folds: int,
) -> dict[str, Any]:
    xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": float(evaluation.models[XGBOOST].named_steps["model"].scale_pos_weight)}
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "versions": (
            f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
            f"xgboost {xgboost.__version__}, pandas {pd.__version__}"
        ),
        "random_state": config.RANDOM_STATE,
        "cv_folds": cv_folds,
        "data": _data_summary(raw, raw_path),
        "split": {
            "train_pct": (1 - config.TEST_SIZE) * 100,
            "test_pct": config.TEST_SIZE * 100,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "churn_train": float(y_train.mean()),
            "churn_test": float(y_test.mean()),
        },
        "hyperparameters": {LOGISTIC: LOGISTIC_PARAMS, XGBOOST: xgb_params},
        "cv": evaluation.cv,
        "test": evaluation.test,
        "selected": selected,
        "selection_rule": SELECTION_RULE,
        "decision_threshold": config.DECISION_THRESHOLD,
        "top_features": {
            name: [(str(f), float(v)) for f, v in series.sort_values(ascending=False).head(TOP_N_FEATURES).items()]
            for name, series in weights.items()
        },
        "segmentation": {**segmentation.info, "profile": segmentation.profile, "descriptions": segmentation.descriptions},
    }


def run_training(
    raw_path: Path | str = config.RAW_PATH,
    models_dir: Path | str = config.MODELS_DIR,
    reports_dir: Path | str = config.REPORTS_DIR,
    processed_dir: Path | str = config.PROCESSED_DIR,
    cv_folds: int = config.CV_FOLDS,
) -> TrainingResult:
    """Flujo completo: limpieza → partición → CV → entrenamiento → evaluación → segmentación → artefactos."""
    raw_path, models_dir = Path(raw_path), Path(models_dir)
    reports_dir, processed_dir = Path(reports_dir), Path(processed_dir)

    raw = load_raw(raw_path)
    clean = standardize_categories(raw)
    X, y = split_features_target(clean)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    logger.info("Datos: %d clientes | train %d | test %d", len(X), len(X_train), len(X_test))

    evaluation = _fit_and_evaluate(X_train, X_test, y_train, y_test, cv_folds)
    selected = select_best_model(evaluation.cv, recall_key="recall_mean", auc_key="roc_auc_mean")
    logger.info("Modelo seleccionado (validación cruzada): %s", selected)

    segmentation = _segment_customers(clean)
    weights = {name: _feature_weights(model, name) for name, model in evaluation.models.items()}

    _persist_data_and_models(selected, evaluation, segmentation, clean, X_test, y_test, models_dir, processed_dir)
    context = _report_context(selected, evaluation, segmentation, raw, raw_path, y_train, y_test, weights, cv_folds)
    _write_reports(selected, evaluation, segmentation, context, y_test, weights, models_dir, reports_dir)

    artifacts = {
        "model": models_dir / config.MODEL_FILE,
        "segmenter": models_dir / config.SEGMENTER_FILE,
        "report": reports_dir / "training_report.md",
    }
    return TrainingResult(selected, evaluation.test, evaluation.cv, int(segmentation.info["k"]), artifacts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de churn y la segmentación.")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    args = parser.parse_args(argv)

    setup_logging()
    result = run_training(cv_folds=args.cv_folds)
    logger.info("Listo. Modelo: %s | Reporte: %s", result.selected_model, result.artifacts["report"])


if __name__ == "__main__":
    main()

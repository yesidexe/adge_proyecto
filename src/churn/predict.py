"""Predicción de churn y segmento para clientes nuevos.

Uso:
    python -m churn.predict --input data/raw/ecommerce_data.csv --output predicciones.csv

Nota de seguridad: `joblib` usa pickle. Cargue únicamente modelos generados por
este proyecto (`python -m churn.train`), nunca archivos de origen desconocido.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from churn import config
from churn.data import load_raw, standardize_categories
from churn.logging_setup import setup_logging
from churn.segment import assign_segments

logger = logging.getLogger(__name__)

RISK_LABELS: tuple[str, ...] = ("Bajo", "Medio", "Alto")


def risk_level(probability: float) -> str:
    """Convierte la probabilidad de churn en una banda de riesgo accionable."""
    if probability >= config.RISK_HIGH_THRESHOLD:
        return "Alto"
    if probability >= config.RISK_MEDIUM_THRESHOLD:
        return "Medio"
    return "Bajo"


def load_artifacts(models_dir: Path | str = config.MODELS_DIR) -> tuple[Pipeline, Pipeline]:
    """Carga el modelo de churn y el segmentador entrenados."""
    models_dir = Path(models_dir)
    model_path, segmenter_path = models_dir / config.MODEL_FILE, models_dir / config.SEGMENTER_FILE
    for path in (model_path, segmenter_path):
        if not path.exists():
            raise FileNotFoundError(
                f"No existe {path}. Entrene primero el modelo con: python -m churn.train"
            )
    return joblib.load(model_path), joblib.load(segmenter_path)


def predict_customers(
    df: pd.DataFrame,
    model: Pipeline,
    segmenter: Pipeline | None = None,
) -> pd.DataFrame:
    """Predice probabilidad, clase, nivel de riesgo y (opcional) segmento por cliente.

    `df` puede traer las columnas originales del dataset; `Churn` y `CustomerID`
    son opcionales (el ID, si existe, se devuelve para identificar a cada cliente).
    """
    if df.empty:
        raise ValueError("No hay clientes para predecir: el DataFrame está vacío.")
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para predecir: {missing}")

    clean = standardize_categories(df).reset_index(drop=True)
    features = clean[list(config.FEATURE_COLUMNS)].copy()
    # Una columna categórica totalmente vacía llega como float64: se fuerza a texto para el One-Hot.
    categorical = list(config.CATEGORICAL_COLUMNS)
    features[categorical] = features[categorical].astype("string")
    probability = model.predict_proba(features)[:, 1]

    result = pd.DataFrame(
        {
            "churn_probability": probability.round(4),
            "churn_prediction": (probability >= config.DECISION_THRESHOLD).astype(int),
            "risk_level": [risk_level(p) for p in probability],
        }
    )
    if segmenter is not None:
        result["segment"] = assign_segments(segmenter, clean)
    if config.ID_COL in clean.columns:
        result.insert(0, config.ID_COL, clean[config.ID_COL])
    return result


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("debe ser un entero positivo")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predice churn y segmento de clientes.")
    parser.add_argument("--input", type=Path, default=config.RAW_PATH, help="CSV con las columnas del dataset.")
    parser.add_argument("--output", type=Path, default=None, help="CSV de salida (opcional).")
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--top", type=_positive_int, default=10, help="Clientes de mayor riesgo a mostrar.")
    args = parser.parse_args(argv)

    setup_logging()
    model, segmenter = load_artifacts(args.models_dir)
    predictions = predict_customers(load_raw(args.input), model, segmenter)

    logger.info("Clientes evaluados: %d", len(predictions))
    logger.info("Distribución de riesgo:\n%s", predictions["risk_level"].value_counts().to_string())
    top = predictions.sort_values("churn_probability", ascending=False).head(args.top)
    logger.info("Top %d de mayor riesgo:\n%s", args.top, top.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        logger.info("Predicciones guardadas en %s", args.output)


if __name__ == "__main__":
    main()

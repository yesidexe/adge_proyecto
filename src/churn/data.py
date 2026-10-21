"""Carga, limpieza y partición de los datos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config


def load_raw(path: Path | str = config.RAW_PATH) -> pd.DataFrame:
    """Lee el CSV original (que nunca se modifica)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {path}")
    return pd.read_csv(path)


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica etiquetas duplicadas (p. ej. `Phone` y `Mobile Phone`). Devuelve una copia.

    Los nulos se conservan a propósito: se imputan dentro del pipeline, usando
    solo los datos de entrenamiento, para evitar fuga de información.
    """
    out = df.copy()
    for column, mapping in config.CATEGORY_FIXES.items():
        if column in out.columns:
            out[column] = out[column].replace(mapping)
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables predictoras (sin `CustomerID`) de la variable objetivo."""
    missing = [c for c in (*config.FEATURE_COLUMNS, config.TARGET) if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")
    return df[list(config.FEATURE_COLUMNS)].copy(), df[config.TARGET].copy()


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Partición 80/20 que conserva la proporción de desertores en ambos conjuntos."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

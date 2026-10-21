"""Rutas y constantes del proyecto (única fuente de verdad)."""

from __future__ import annotations

from pathlib import Path

# Raíz del repositorio (este archivo está en src/churn/).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "ecommerce_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODEL_FILE = "churn_model.joblib"
SEGMENTER_FILE = "segmenter.joblib"

# Reproducibilidad y partición (igual que los notebooks 01 y 02).
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "Churn"
ID_COL = "CustomerID"

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",  # sic: así está escrito en el dataset original
    "MaritalStatus",
)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
)

FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Etiquetas duplicadas del dataset original que se unifican en la limpieza.
CATEGORY_FIXES: dict[str, dict[str, str]] = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}

# Umbral de decisión del clasificador y bandas de riesgo (heurísticas, no calibradas).
DECISION_THRESHOLD = 0.50
RISK_MEDIUM_THRESHOLD = 0.40
RISK_HIGH_THRESHOLD = 0.70

import numpy as np
import pandas as pd
import pytest


def build_raw_df() -> pd.DataFrame:
    """Dataset sintético con el mismo esquema que `ecommerce_data.csv` (con nulos y categorías sucias)."""
    rng = np.random.default_rng(0)
    n = 400
    complain = rng.integers(0, 2, n)
    tenure = rng.integers(0, 30, n).astype(float)
    churn_logit = -1.0 + 1.8 * complain - 0.08 * tenure + rng.normal(0, 0.5, n)
    churn = (rng.random(n) < 1 / (1 + np.exp(-churn_logit))).astype(int)

    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + n),
            "Churn": churn,
            "Tenure": tenure,
            "PreferredLoginDevice": rng.choice(["Mobile Phone", "Phone", "Computer"], n),
            "CityTier": rng.integers(1, 4, n),
            "WarehouseToHome": rng.integers(5, 35, n).astype(float),
            "PreferredPaymentMode": rng.choice(["Debit Card", "CC", "COD", "UPI", "E wallet"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "HourSpendOnApp": rng.integers(0, 5, n).astype(float),
            "NumberOfDeviceRegistered": rng.integers(1, 6, n),
            "PreferedOrderCat": rng.choice(["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion"], n),
            "SatisfactionScore": rng.integers(1, 6, n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
            "NumberOfAddress": rng.integers(1, 10, n),
            "Complain": complain,
            "OrderAmountHikeFromlastYear": rng.integers(11, 26, n).astype(float),
            "CouponUsed": rng.integers(0, 6, n).astype(float),
            "OrderCount": rng.integers(1, 12, n).astype(float),
            "DaySinceLastOrder": rng.integers(0, 20, n).astype(float),
            "CashbackAmount": rng.normal(180, 40, n).round(2),
        }
    )
    # Introducir nulos en columnas numéricas, como en el dataset real.
    for col in ["Tenure", "WarehouseToHome", "DaySinceLastOrder"]:
        df.loc[rng.choice(n, 20, replace=False), col] = np.nan
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return build_raw_df()


@pytest.fixture(scope="module")
def raw_df_shared() -> pd.DataFrame:
    """Misma data, compartida por módulo para no reentrenar en cada prueba."""
    return build_raw_df()

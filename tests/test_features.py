import numpy as np

from churn import config
from churn.data import split_features_target, standardize_categories
from churn.features import build_preprocessor


def _features(raw_df):
    X, _ = split_features_target(standardize_categories(raw_df))
    return X


def test_preprocessor_outputs_no_nulls_even_with_missing_inputs(raw_df):
    X = _features(raw_df)
    assert X.isna().any().any()

    out = build_preprocessor(scale=True).fit_transform(X)

    assert not np.isnan(out).any()


def test_preprocessor_scaling_centers_numeric_columns(raw_df):
    X = _features(raw_df)

    out = build_preprocessor(scale=True).fit_transform(X)

    n_num = len(config.NUMERIC_COLUMNS)
    assert np.allclose(out[:, :n_num].mean(axis=0), 0, atol=1e-6)


def test_preprocessor_without_scaling_keeps_original_scale(raw_df):
    X = _features(raw_df)

    out = build_preprocessor(scale=False).fit_transform(X)

    tenure_idx = config.NUMERIC_COLUMNS.index("Tenure")
    assert out[:, tenure_idx].max() == X["Tenure"].max()


def test_preprocessor_ignores_unseen_categories_at_predict_time(raw_df):
    X = _features(raw_df)
    pre = build_preprocessor(scale=True).fit(X)
    new = X.head(3).copy()
    new["PreferredPaymentMode"] = "Bitcoin"

    out = pre.transform(new)

    assert out.shape[0] == 3


def test_preprocessor_imputation_is_fitted_only_on_training_data(raw_df):
    X = _features(raw_df)
    train, other = X.iloc[:200], X.iloc[200:].copy()
    other["Tenure"] = np.nan
    pre = build_preprocessor(scale=False).fit(train)

    out = pre.transform(other)

    tenure_idx = config.NUMERIC_COLUMNS.index("Tenure")
    assert np.allclose(out[:, tenure_idx], train["Tenure"].median())

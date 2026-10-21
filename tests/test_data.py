import pandas as pd
import pytest

from churn import config
from churn.data import (
    load_raw,
    split_features_target,
    standardize_categories,
    train_test_split_stratified,
)


def test_standardize_categories_unifies_duplicated_labels(raw_df):
    clean = standardize_categories(raw_df)

    assert set(clean["PreferredLoginDevice"]) <= {"Mobile Phone", "Computer"}
    assert "CC" not in set(clean["PreferredPaymentMode"])
    assert "COD" not in set(clean["PreferredPaymentMode"])
    assert "Mobile" not in set(clean["PreferedOrderCat"])


def test_standardize_categories_does_not_mutate_input(raw_df):
    snapshot = raw_df.copy(deep=True)

    standardize_categories(raw_df)

    pd.testing.assert_frame_equal(raw_df, snapshot)


def test_standardize_categories_keeps_nulls_for_later_imputation(raw_df):
    clean = standardize_categories(raw_df)

    assert clean["Tenure"].isna().sum() == raw_df["Tenure"].isna().sum()


def test_split_features_target_drops_id_and_target(raw_df):
    X, y = split_features_target(raw_df)

    assert config.TARGET not in X.columns
    assert config.ID_COL not in X.columns
    assert list(X.columns) == list(config.FEATURE_COLUMNS)
    assert y.name == config.TARGET


def test_split_features_target_fails_fast_on_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Tenure"):
        split_features_target(raw_df.drop(columns=["Tenure"]))


def test_train_test_split_is_stratified_and_reproducible(raw_df):
    X, y = split_features_target(standardize_categories(raw_df))

    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y)
    X_tr2, _, _, _ = train_test_split_stratified(X, y)

    assert len(X_te) == pytest.approx(len(X) * config.TEST_SIZE, abs=1)
    assert y_tr.mean() == pytest.approx(y_te.mean(), abs=0.02)
    assert list(X_tr.index) == list(X_tr2.index)


def test_load_raw_real_dataset_has_expected_shape():
    df = load_raw()

    assert df.shape == (5630, 20)
    assert set(df[config.TARGET].unique()) == {0, 1}


def test_load_raw_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="no_existe.csv"):
        load_raw(tmp_path / "no_existe.csv")

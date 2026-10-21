import pytest

from churn.data import split_features_target, standardize_categories, train_test_split_stratified
from churn.evaluate import compute_metrics
from churn.modeling import MODEL_NAMES, build_models


def test_compute_metrics_known_values():
    y_true = [1, 1, 0, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 1, 1]
    y_prob = [0.9, 0.4, 0.2, 0.1, 0.6, 0.8]

    m = compute_metrics(y_true, y_pred, y_prob)

    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 2
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_build_models_returns_both_models(raw_df):
    X, y = split_features_target(standardize_categories(raw_df))

    models = build_models(y)

    assert set(models) == set(MODEL_NAMES)


def test_models_fit_and_predict_probabilities(raw_df):
    X, y = split_features_target(standardize_categories(raw_df))
    X_tr, X_te, y_tr, _ = train_test_split_stratified(X, y)

    for name, model in build_models(y_tr).items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        assert proba.shape == (len(X_te),), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


def test_models_beat_random_on_synthetic_signal(raw_df):
    X, y = split_features_target(standardize_categories(raw_df))
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X, y)

    for name, model in build_models(y_tr).items():
        model.fit(X_tr, y_tr)
        m = compute_metrics(y_te, model.predict(X_te), model.predict_proba(X_te)[:, 1])
        assert m["roc_auc"] > 0.6, name


def test_select_best_model_prefers_recall_then_auc():
    from churn.evaluate import select_best_model

    metrics = {
        "A": {"recall": 0.90, "roc_auc": 0.80},
        "B": {"recall": 0.90, "roc_auc": 0.85},
        "C": {"recall": 0.70, "roc_auc": 0.99},
    }

    assert select_best_model(metrics) == "B"


def test_select_best_model_rejects_empty_input():
    from churn.evaluate import select_best_model

    with pytest.raises(ValueError):
        select_best_model({})


def test_build_models_requires_both_classes():
    import pandas as pd

    with pytest.raises(ValueError, match="ambas clases"):
        build_models(pd.Series([0, 0, 0]))


def test_select_best_model_accepts_custom_metric_keys():
    from churn.evaluate import select_best_model

    cv = {"A": {"recall_mean": 0.80, "roc_auc_mean": 0.90}, "B": {"recall_mean": 0.85, "roc_auc_mean": 0.80}}

    assert select_best_model(cv, recall_key="recall_mean", auc_key="roc_auc_mean") == "B"

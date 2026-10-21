import json

import pandas as pd
import pytest

from churn.predict import RISK_LABELS, load_artifacts, predict_customers, risk_level
from churn.train import run_training


@pytest.fixture(scope="module")
def trained(raw_df_shared, tmp_path_factory):
    root = tmp_path_factory.mktemp("trained")
    raw_path = root / "raw.csv"
    raw_df_shared.to_csv(raw_path, index=False)
    result = run_training(
        raw_path=raw_path,
        models_dir=root / "models",
        reports_dir=root / "reports",
        processed_dir=root / "processed",
        cv_folds=3,
    )
    return result, root


def test_run_training_writes_all_artifacts(trained):
    _, root = trained

    expected = [
        root / "models" / "churn_model.joblib",
        root / "models" / "segmenter.joblib",
        root / "models" / "model_card.json",
        root / "reports" / "metrics.json",
        root / "reports" / "training_report.md",
        root / "reports" / "figures" / "roc_curves.png",
        root / "reports" / "figures" / "confusion_matrices.png",
        root / "reports" / "figures" / "feature_importance.png",
        root / "reports" / "figures" / "segments.png",
        root / "processed" / "test_predictions.csv",
        root / "processed" / "customer_segments.csv",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing


def test_run_training_metrics_cover_both_models_and_cv(trained):
    result, root = trained
    metrics = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))

    assert set(metrics["test"]) == {"Regresion Logistica", "XGBoost"}
    assert metrics["selected_model"] in metrics["test"]
    assert set(metrics["cv"]["XGBoost"]) >= {"recall_mean", "roc_auc_mean"}
    assert result.selected_model == metrics["selected_model"]


def test_training_report_explains_how_the_model_was_trained(trained):
    _, root = trained
    text = (root / "reports" / "training_report.md").read_text(encoding="utf-8")

    for section in ("Datos", "Preprocesamiento", "Modelos", "Validación cruzada", "Resultados en prueba", "Segmentación"):
        assert section in text


def test_predict_customers_returns_probability_label_risk_and_segment(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")

    out = predict_customers(raw_df.head(10), model, segmenter)

    assert list(out["CustomerID"]) == list(raw_df["CustomerID"].head(10))
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["churn_prediction"]) <= {0, 1}
    assert set(out["risk_level"]) <= set(RISK_LABELS)
    assert out["segment"].notna().all()


def test_predict_customers_accepts_rows_without_target_and_id(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")

    out = predict_customers(raw_df.drop(columns=["Churn", "CustomerID"]).head(3), model, segmenter)

    assert len(out) == 3
    assert "CustomerID" not in out.columns


def test_predict_customers_rejects_missing_columns(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")

    with pytest.raises(ValueError, match="Tenure"):
        predict_customers(raw_df.drop(columns=["Tenure"]), model, segmenter)


def test_risk_level_bands():
    assert risk_level(0.05) == "Bajo"
    assert risk_level(0.45) == "Medio"
    assert risk_level(0.95) == "Alto"
    assert risk_level(0.0) == "Bajo" and risk_level(1.0) == "Alto"


def test_load_artifacts_fails_clearly_when_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m churn.train"):
        load_artifacts(tmp_path)


def test_test_predictions_csv_has_probabilities_for_each_model(trained):
    _, root = trained
    preds = pd.read_csv(root / "processed" / "test_predictions.csv")

    assert {"CustomerID", "Churn", "prob_Regresion Logistica", "prob_XGBoost"} <= set(preds.columns)


def test_predict_cli_writes_predictions_csv(trained):
    from churn.predict import main

    _, root = trained
    output = root / "out" / "pred.csv"

    main(["--input", str(root / "raw.csv"), "--models-dir", str(root / "models"), "--output", str(output), "--top", "3"])

    written = pd.read_csv(output)
    assert {"CustomerID", "churn_probability", "risk_level", "segment"} <= set(written.columns)
    assert len(written) == 400


def test_predict_customers_handles_all_null_categorical_column(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")
    one = raw_df.head(1).assign(Gender=float("nan"))

    out = predict_customers(one, model, segmenter)

    assert len(out) == 1 and 0 <= out.loc[0, "churn_probability"] <= 1


def test_predict_customers_handles_unseen_category_end_to_end(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")

    out = predict_customers(raw_df.head(3).assign(PreferredPaymentMode="Bitcoin"), model, segmenter)

    assert out["churn_probability"].between(0, 1).all()


def test_predict_customers_rejects_empty_input(trained, raw_df):
    _, root = trained
    model, segmenter = load_artifacts(root / "models")

    with pytest.raises(ValueError, match="vac"):
        predict_customers(raw_df.head(0), model, segmenter)


def test_predict_cli_rejects_non_positive_top(trained):
    from churn.predict import main

    _, root = trained

    with pytest.raises(SystemExit):
        main(["--input", str(root / "raw.csv"), "--models-dir", str(root / "models"), "--top", "-1"])


def test_selected_model_is_chosen_with_cross_validation_not_test(trained):
    result, _ = trained

    best_by_cv = max(result.cv_metrics, key=lambda n: (result.cv_metrics[n]["recall_mean"], result.cv_metrics[n]["roc_auc_mean"]))

    assert result.selected_model == best_by_cv


def test_test_metrics_use_the_configured_decision_threshold(raw_df, tmp_path, monkeypatch):
    from churn import config

    monkeypatch.setattr(config, "DECISION_THRESHOLD", 0.99)
    raw_path = tmp_path / "raw.csv"
    raw_df.to_csv(raw_path, index=False)

    result = run_training(raw_path, tmp_path / "m", tmp_path / "r", tmp_path / "p", cv_folds=3)

    # Con un umbral extremo casi nada se clasifica como churn: el recall cae respecto al umbral 0,5.
    assert all(m["recall"] < 0.5 for m in result.test_metrics.values())


def test_training_report_uses_actual_source_path_and_scale_pos_weight(trained):
    _, root = trained
    text = (root / "reports" / "training_report.md").read_text(encoding="utf-8")
    card = json.loads((root / "models" / "model_card.json").read_text(encoding="utf-8"))

    assert "raw.csv" in text
    assert "retenidos/desertores (train)" not in text
    assert isinstance(card["hyperparameters"].get("scale_pos_weight", 1.0), float)

import pandas as pd
import pytest

from churn.data import standardize_categories
from churn.segment import (
    SEGMENT_FEATURES,
    assign_segments,
    describe_segments,
    fit_segmenter,
    profile_segments,
)


def test_fit_segmenter_picks_k_within_range_and_labels_every_row(raw_df):
    df = standardize_categories(raw_df)

    segmenter, info = fit_segmenter(df, k_range=range(2, 5))
    labels = assign_segments(segmenter, df)

    assert info["k"] in range(2, 5)
    assert len(labels) == len(df)
    assert set(labels) == set(range(info["k"]))
    assert set(info["silhouette_by_k"]) == {2, 3, 4}


def test_assign_segments_handles_missing_values(raw_df):
    df = standardize_categories(raw_df)
    segmenter, _ = fit_segmenter(df, k_range=range(2, 4))

    labels = assign_segments(segmenter, df.assign(Tenure=float("nan")))

    assert len(labels) == len(df)


def test_profile_segments_reports_size_and_churn_rate(raw_df):
    df = standardize_categories(raw_df)
    segmenter, info = fit_segmenter(df, k_range=range(2, 4))
    labels = assign_segments(segmenter, df)

    profile = profile_segments(df, labels)

    assert profile["clientes"].sum() == len(df)
    assert profile["tasa_churn"].between(0, 1).all()
    assert set(SEGMENT_FEATURES) <= set(profile.columns)
    assert len(profile) == info["k"]


def test_describe_segments_flags_features_far_from_average():
    profile = pd.DataFrame(
        {
            "clientes": [50, 50],
            "Tenure": [2.0, 18.0],
            "DaySinceLastOrder": [5.0, 5.0],
            "OrderCount": [3.0, 3.0],
            "CashbackAmount": [100.0, 100.0],
        },
        index=pd.Index([0, 1], name="segmento"),
    )

    text = describe_segments(profile)

    assert "antigüedad menor" in text[0]
    assert "antigüedad mayor" in text[1]
    assert "órdenes" not in text[0]


def test_describe_segments_reports_average_profile_when_no_deviation():
    profile = pd.DataFrame(
        {"clientes": [10], "Tenure": [5.0], "DaySinceLastOrder": [5.0], "OrderCount": [5.0], "CashbackAmount": [5.0]},
        index=pd.Index([0], name="segmento"),
    )

    assert describe_segments(profile)[0] == "perfil cercano al promedio"


def test_fit_segmenter_rejects_missing_feature_columns(raw_df):
    with pytest.raises(ValueError, match="CashbackAmount"):
        fit_segmenter(raw_df.drop(columns=["CashbackAmount"]))


def test_profile_segments_matches_hand_computed_values():
    import numpy as np

    df = pd.DataFrame(
        {
            "Tenure": [1.0, 3.0, 10.0, 20.0],
            "DaySinceLastOrder": [1.0, 1.0, 5.0, 5.0],
            "OrderCount": [2.0, 4.0, 6.0, 8.0],
            "CashbackAmount": [100.0, 100.0, 200.0, 200.0],
            "Churn": [1, 0, 0, 0],
        }
    )

    profile = profile_segments(df, np.array([0, 0, 1, 1]))

    assert profile.loc[0, "clientes"] == 2 and profile.loc[1, "clientes"] == 2
    assert profile.loc[0, "tasa_churn"] == 0.5 and profile.loc[1, "tasa_churn"] == 0.0
    assert profile.loc[0, "Tenure"] == 2.0 and profile.loc[1, "Tenure"] == 15.0

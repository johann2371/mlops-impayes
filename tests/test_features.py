"""Tests de src/features/build_features.py (tâche 3).

Les données ci-dessous sont **fabriquées pour les tests**, avec les noms
de colonnes du jeu réel. Elles ne servent qu'à vérifier le code, pas à
tirer une conclusion sur le jeu de données.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{i}" for i in range(n)],
            "phone_number": 2347000000000 + np.arange(n),
            "operator": rng.choice(["MTN", "Airtel", "Glo", "9mobile"], n),
            "assessment_date": "2025-09-30",
            "credit_score": rng.integers(300, 1001, n),
            "credit_rating": rng.choice(
                ["poor", "fair", "good", "excellent"], n
            ),
            "tenure_months": rng.integers(1, 121, n),
            "avg_monthly_spend_ngn": rng.uniform(500, 50000, n).round(2),
            "recharge_frequency_per_month": rng.integers(1, 31, n),
            "payment_consistency_score": rng.integers(0, 101, n) / 100,
            "late_payment_count": rng.integers(0, 11, n),
            "account_type": rng.choice(
                ["prepaid", "postpaid"], n, p=[0.85, 0.15]
            ),
            "loan_eligibility": rng.uniform(size=n) > 0.3,
            "max_loan_amount_ngn": rng.uniform(0, 150000, n).round(2),
            "default_risk": rng.choice(
                ["low", "medium", "high"], n, p=[0.55, 0.23, 0.22]
            ),
        }
    )


def test_load_raw_detects_missing_columns(tmp_path, raw_df):
    path = tmp_path / "x.csv"
    raw_df.drop(columns=["tenure_months"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="tenure_months"):
        bf.load_raw(path)


def test_load_raw_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        bf.load_raw(tmp_path / "absent.csv")


def test_clean_drops_identifiers_and_constants(raw_df):
    out = bf.clean(raw_df)
    for col in bf.ID_COLUMNS + bf.CONSTANT_COLUMNS:
        assert col not in out.columns


def test_clean_rejects_unknown_target_value(raw_df):
    raw_df.loc[0, "default_risk"] = "critical"
    with pytest.raises(ValueError, match="critical"):
        bf.clean(raw_df)


def test_encode_decode_target_roundtrip():
    y = pd.Series(["low", "high", "medium", "low"])
    codes = bf.encode_target(y)
    assert codes.tolist() == [0, 2, 1, 0]
    assert bf.decode_target(codes) == y.tolist()


def test_split_is_stratified_and_reproducible(raw_df):
    a = bf.split_data(raw_df, seed=1)
    b = bf.split_data(raw_df, seed=1)
    assert a[0].index.equals(b[0].index)
    full = (
        bf.encode_target(raw_df["default_risk"])
        .value_counts(normalize=True)
        .sort_index()
    )
    test = a[3].value_counts(normalize=True).sort_index()
    assert np.allclose(full.values, test.values, atol=0.02)
    assert len(a[0]) + len(a[1]) == len(raw_df)


def test_split_has_no_row_overlap(raw_df):
    X_train, X_test, _, _ = bf.split_data(raw_df)
    assert set(X_train.index).isdisjoint(X_test.index)


@pytest.mark.parametrize("feature_set", ["core", "with_scoring"])
@pytest.mark.parametrize("add_ratios", [False, True])
def test_pipeline_output_is_numeric_without_nan(
    raw_df, feature_set, add_ratios
):
    X_train, X_test, _, _ = bf.split_data(raw_df, feature_set)
    pipe = bf.build_pipeline(feature_set, add_ratios).fit(X_train)
    Xt = pipe.transform(X_test)
    assert Xt.shape[0] == len(X_test)
    assert not Xt.isna().any().any()
    assert np.isfinite(Xt.to_numpy()).all()
    assert all(pd.api.types.is_numeric_dtype(t) for t in Xt.dtypes)


def test_scaler_is_fitted_on_train_only(raw_df):
    X_train, X_test, _, _ = bf.split_data(raw_df)
    pipe = bf.build_pipeline("core").fit(X_train)
    scaler = (
        pipe.named_steps["preprocess"]
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    assert np.isclose(scaler.mean_[0], X_train["tenure_months"].mean())
    assert not np.isclose(
        scaler.mean_[0], raw_df["tenure_months"].mean(), atol=1e-12
    )


def test_unseen_category_does_not_crash(raw_df):
    X_train, X_test, _, _ = bf.split_data(raw_df)
    pipe = bf.build_pipeline("core").fit(X_train)
    X_new = X_test.head(3).copy()
    X_new["operator"] = "NouvelOperateur"
    assert pipe.transform(X_new).shape[0] == 3


def test_missing_values_are_imputed(raw_df):
    X_train, X_test, _, _ = bf.split_data(raw_df)
    X_train = X_train.copy()
    X_train.iloc[:20, X_train.columns.get_loc("avg_monthly_spend_ngn")] = (
        np.nan
    )
    pipe = bf.build_pipeline("core").fit(X_train)
    assert not pipe.transform(X_train).isna().any().any()


def test_ratio_features_have_no_division_error():
    X = pd.DataFrame(
        {
            "avg_monthly_spend_ngn": [1000.0],
            "recharge_frequency_per_month": [0],
            "late_payment_count": [3],
            "tenure_months": [0],
        }
    )
    out = bf.add_ratio_features(X)
    assert np.isfinite(out[bf.RATIO_COLUMNS].to_numpy()).all()


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError):
        bf.build_pipeline("inconnu")


def test_pipeline_is_serializable(raw_df, tmp_path):
    X_train, X_test, _, _ = bf.split_data(raw_df, "with_scoring")
    pipe = bf.build_pipeline("with_scoring", add_ratios=True).fit(X_train)
    joblib.dump(pipe, tmp_path / "pipe.joblib")
    reloaded = joblib.load(tmp_path / "pipe.joblib")
    assert np.allclose(
        pipe.transform(X_test).to_numpy(),
        reloaded.transform(X_test).to_numpy(),
    )


def test_core_set_excludes_scoring_columns(raw_df):
    X_train, _, _, _ = bf.split_data(raw_df, "core")
    for col in [
        "credit_score",
        "credit_rating",
        "loan_eligibility",
        "max_loan_amount_ngn",
    ]:
        assert col not in X_train.columns

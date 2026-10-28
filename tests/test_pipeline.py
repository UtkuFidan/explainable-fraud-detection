"""Offline tests on small fixtures: no dataset download, a few seconds of training."""

import joblib
import numpy as np
import pytest

from fraud_xai import metrics as M
from fraud_xai.data import (
    BAF_CATEGORICAL,
    baf_features,
    baf_task,
    make_synthetic_baf,
    ulb_task,
)
from fraud_xai.explain import reason_codes
from fraud_xai.pipeline import Config, run_experiment
from fraud_xai.report import build_report

CFG = Config(
    trials={"lightgbm": 2, "xgboost": 2, "catboost": 2},
    tune_rows={"lightgbm": 3000, "xgboost": 3000, "catboost": 3000},
    n_jobs=2,
    n_boot=30,
)


@pytest.fixture(scope="module")
def synthetic():
    return make_synthetic_baf(n=8000)


@pytest.fixture(scope="module")
def trained(synthetic, tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    return out, run_experiment(baf_task(synthetic, "synthetic"), out, CFG)


def test_baf_split_is_temporal(synthetic):
    t = baf_task(synthetic)
    m = t.period
    assert m[t.train].max() == 4
    assert set(m[t.tune]) == set(m[t.cal]) == {5}
    assert set(m[t.test]) == {6, 7}
    assert not set(t.tune) & set(t.cal)


def test_features_handle_sentinels_and_categoricals(synthetic):
    X = baf_features(synthetic)
    assert "device_fraud_count" not in X and "fraud_bool" not in X and "month" not in X
    assert (synthetic["prev_address_months_count"] == -1).any()
    assert X["prev_address_months_count"].isna().any()
    assert (X["intended_balcon_amount"].dropna() >= 0).all()
    assert list(X["device_os"].cat.categories) == BAF_CATEGORICAL["device_os"]
    assert {"credit_limit_to_income", "velocity_6h_to_24h"} <= set(X)


def test_ulb_split_is_chronological():
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(size=(500, 30)), columns=[*[f"V{i}" for i in range(1, 29)], "x", "y"]
    )
    df = df.rename(columns={"x": "Amount", "y": "Class"})
    df["Class"] = (rng.random(500) < 0.05).astype(int)
    df["Time"] = rng.permutation(500).astype(float)
    t = ulb_task(df)
    time = df.sort_values("Time")["Time"].to_numpy()
    assert time[t.train].max() < time[t.tune].min() and time[t.cal].max() < time[t.test].min()


def test_pipeline_outputs(trained):
    _, m = trained
    assert set(m["test"]) == {"logistic_regression", "lightgbm", "xgboost", "catboost", "blend"}
    for k, v in m["test"].items():
        for name in ("tpr_at_5fpr", "pr_auc", "recall_at_budget"):
            assert 0 <= v[name]["lo"] <= v[name]["hi"] <= 1, (k, name)
    assert m["test"][m["deployed"]]["tpr_at_5fpr"]["value"] > 0.2
    assert set(m["by_month"]) == {6, 7}
    assert m["calibration"]["chosen"] in {"platt", "isotonic"}


def test_group_thresholds_equalise_fpr(trained):
    _, m = trained
    f = m["fairness"]["deployed"]
    g, h = f["global_threshold"], f["group_thresholds"]
    assert abs(h["fpr_gap"]) <= abs(g["fpr_gap"]) + 0.02
    assert f["thresholds"]["over_50"] != f["thresholds"]["under_50"]


def test_reason_codes_use_real_names(trained):
    out, _ = trained
    b = joblib.load(out / "bundle.joblib")
    feats = list(b["reference"].columns)
    X = b["test"][feats].head(6)
    codes = reason_codes(b["model"].shap(X), X, b["reference"], top=3)
    assert len(codes) == 6
    for cs in codes:
        assert len({c["feature"] for c in cs}) == 3
        assert all(c["feature"] in feats for c in cs)
        mags = [abs(c["shap"]) for c in cs]
        assert mags == sorted(mags, reverse=True)
        assert all(c["percentile"] is None or 0 <= c["percentile"] <= 1 for c in cs)


def test_report_writes_case_file(trained, tmp_path):
    out, _ = trained
    path = build_report(out, tmp_path / "docs", tmp_path / "reports", top_n=3, max_global=200)
    text = path.read_text()
    assert "Review checklist" in text and "- [ ]" in text
    for fig in ("shap_summary", "calibration", "pr_curves", "drift", "fairness"):
        assert (tmp_path / "docs" / f"{fig}.png").exists()


def test_metrics_basic():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert M.pr_auc(y, s) == pytest.approx(0.5 + (2 / 3) * 0.5)
    assert M.recall_at_frac(y, s, 0.2) == 0.5
    assert M.recall_at_frac(np.zeros(3), np.ones(3), 0.5) == 0.0
    assert M.tpr_at_fpr(y, s, 0.4) == 1.0
    assert M.tpr_at_fpr(y, s, 0.3) == 0.5


def test_bootstrap_point_matches_and_brackets():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    s = rng.normal(size=4000) + 1.5 * y
    r = M.bootstrap_metrics(y, s, 0.05, n_boot=200)
    assert r["tpr_at_5fpr"]["value"] == pytest.approx(M.tpr_at_fpr(y, s))
    for v in r.values():
        assert v["lo"] <= v["value"] <= v["hi"]
    # weighted implementation with unit weights agrees with sklearn
    order, ys = M._sorted_curves(y, s)
    w = np.ones(len(ys))
    assert M._w_pr_auc(ys, w) == pytest.approx(M.pr_auc(y, s), abs=1e-6)
    assert M._w_tpr(ys, w, 0.05) == pytest.approx(M.tpr_at_fpr(y, s), abs=2e-3)


def test_group_rates_and_threshold():
    rng = np.random.default_rng(2)
    n = 20000
    older = rng.random(n) < 0.2
    y = (rng.random(n) < np.where(older, 0.04, 0.01)).astype(int)
    s = rng.normal(size=n) + 2 * y + 0.8 * older
    thr = M.threshold_at_fpr(y, s)
    r = M.group_rates(y, s, older, thr, thr)
    assert r["overall"]["fpr"] == pytest.approx(0.05, abs=0.005)
    assert r["fpr_ratio_over_under"] > 1.5  # shifted scores push the older group's FPR up
    t_y, t_o = (M.threshold_at_fpr(y[g], s[g]) for g in (~older, older))
    r2 = M.group_rates(y, s, older, t_y, t_o)
    assert r2["over_50"]["fpr"] == pytest.approx(r2["under_50"]["fpr"], abs=0.01)


def test_calibration_and_psi():
    rng = np.random.default_rng(3)
    m = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-(m * 1.5 - 3)))).astype(int)
    cal, scores = M.pick_calibrator(m, y)
    assert cal.kind in scores
    p = cal.predict(m)
    assert M.expected_calibration_error(y, p) < M.expected_calibration_error(
        y, 1 / (1 + np.exp(-m))
    )
    a = rng.normal(size=5000)
    assert M.psi(a, a + rng.normal(scale=0.01, size=5000)) < 0.01
    assert M.psi(a, a + 1.0) > 0.25

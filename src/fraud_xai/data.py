"""Datasets: Bank Account Fraud (BAF, primary), ULB credit-card fraud (benchmark), and a fixture.

Both datasets are turned into a ``Task``: a feature frame, labels and positional index arrays for
the train / tune / calibration / test windows. Everything downstream works on a ``Task``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---- Bank Account Fraud (Jesus et al., NeurIPS 2022) -------------------------------------------
BAF_TARGET = "fraud_bool"
BAF_MONTH = "month"
BAF_AGE = "customer_age"
BAF_CATEGORICAL = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}
# -1 is the documented "missing" code in these columns
BAF_MINUS_ONE_MISSING = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]
BAF_VARIANTS = ["Base", "Variant I", "Variant II", "Variant III", "Variant IV", "Variant V"]
BAF_TRAIN_MONTHS = range(0, 5)  # fit; month 5 is held back for tuning and calibration
BAF_VALID_MONTH = 5
BAF_TEST_MONTHS = (6, 7)
OLDER_AGE = 50

# ---- ULB credit-card fraud (OpenML 1597) -------------------------------------------------------
ULB_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
ULB_TARGET = "Class"
ULB_TIME = "Time"
OPENML_ID = 1597
ORIGINAL_SPAN_SECONDS = 172_792  # span of the original dataset's Time column


@dataclass
class Task:
    """A modelling problem: features, labels and the four windows (positional indices)."""

    name: str
    X: pd.DataFrame
    y: np.ndarray
    train: np.ndarray
    tune: np.ndarray  # early stopping and Optuna objective
    cal: np.ndarray  # calibration, blending, model selection and alert thresholds
    test: np.ndarray
    cat_cols: list[str] = field(default_factory=list)
    period: np.ndarray | None = None  # month of each row (BAF) for the drift breakdown
    older: np.ndarray | None = None  # protected-group flag (customer_age >= 50)
    budget_frac: float = 0.02  # review capacity as a share of applications
    tune_metric: str = "tpr_at_5fpr"  # Optuna objective on the tuning window

    def part(self, which: str) -> tuple[pd.DataFrame, np.ndarray]:
        idx = getattr(self, which)
        return self.X.iloc[idx], self.y[idx]


def _split_validation(idx: np.ndarray, seed: int, tune_share: float = 0.6):
    """Random split of the validation window into a tuning part and a calibration part."""
    perm = np.random.default_rng(seed).permutation(idx)
    k = int(len(perm) * tune_share)
    return np.sort(perm[:k]), np.sort(perm[k:])


def load_baf(data_dir: str | Path = "data", variant: str = "Base") -> pd.DataFrame:
    path = Path(data_dir) / "baf" / f"{variant}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle dataset "
            "'sgpjesus/bank-account-fraud-dataset-neurips-2022' into data/baf/ "
            "(see README, 'Data')."
        )
    return pd.read_csv(path)


def baf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: sentinel codes become NaN, categoricals get fixed levels, plus ratios."""
    X = df.drop(columns=[BAF_TARGET, BAF_MONTH, "device_fraud_count"], errors="ignore").copy()
    for c in BAF_MINUS_ONE_MISSING:
        X[c] = X[c].astype(float).where(X[c] != -1)
    X["intended_balcon_amount"] = X["intended_balcon_amount"].where(
        X["intended_balcon_amount"] >= 0
    )  # negative values mean "not applicable" in the source documentation
    eps = 1e-6
    X["credit_limit_to_income"] = X["proposed_credit_limit"] / (X["income"] + eps)
    X["velocity_6h_to_24h"] = X["velocity_6h"] / (X["velocity_24h"] + eps)
    X["velocity_24h_to_4w"] = X["velocity_24h"] / (X["velocity_4w"] + eps)
    X["zip_count_per_branch"] = X["zip_count_4w"] / (X["bank_branch_count_8w"] + 1)
    X["address_share_of_life"] = X["current_address_months_count"] / (X[BAF_AGE] * 12)
    for c, levels in BAF_CATEGORICAL.items():
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


def baf_task(df: pd.DataFrame, name: str = "BAF Base", seed: int = 0) -> Task:
    """Temporal split as BAF intends: months 0-4 fit, month 5 tunes/calibrates, months 6-7 test."""
    month = df[BAF_MONTH].to_numpy()
    train = np.flatnonzero(np.isin(month, list(BAF_TRAIN_MONTHS)))
    valid = np.flatnonzero(month == BAF_VALID_MONTH)
    test = np.flatnonzero(np.isin(month, BAF_TEST_MONTHS))
    tune, cal = _split_validation(valid, seed)
    X = baf_features(df)
    return Task(
        name=name,
        X=X,
        y=df[BAF_TARGET].to_numpy(),
        train=train,
        tune=tune,
        cal=cal,
        test=test,
        cat_cols=list(BAF_CATEGORICAL),
        period=month,
        older=(df[BAF_AGE] >= OLDER_AGE).to_numpy(),
        budget_frac=0.02,
    )


def load_ulb(cache_dir: str | Path = "data") -> pd.DataFrame:
    """ULB credit-card fraud data (OpenML 1597, ODbL), downloaded once into ``cache_dir``."""
    cache = Path(cache_dir) / "creditcard.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        data_id=OPENML_ID, as_frame=True, data_home=str(Path(cache_dir) / "openml")
    )
    df = bunch.frame.assign(**{ULB_TARGET: lambda d: d[ULB_TARGET].astype(str).astype(int)})
    # OpenML drops the original ``Time`` column but keeps row order, so spread rows evenly over
    # the original 48 h. Only the order matters for the chronological split.
    df = df.assign(**{ULB_TIME: np.arange(len(df)) / len(df) * ORIGINAL_SPAN_SECONDS})
    df = df[[ULB_TIME, *ULB_FEATURES, ULB_TARGET]].astype(float).astype({ULB_TARGET: int})
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, compression="gzip")
    return df


def ulb_task(df: pd.DataFrame, seed: int = 0) -> Task:
    """Chronological 70 / 15 / 15; the middle 15% is split into tuning and calibration parts."""
    df = df.sort_values(ULB_TIME, kind="stable").reset_index(drop=True)
    n = len(df)
    a, b = int(n * 0.70), int(n * 0.85)
    tune, cal = _split_validation(np.arange(a, b), seed)
    return Task(
        name="ULB 2013",
        X=df[ULB_FEATURES],
        y=df[ULB_TARGET].to_numpy(),
        train=np.arange(a),
        tune=tune,
        cal=cal,
        test=np.arange(b, n),
        budget_frac=0.005,
        tune_metric="pr_auc",  # TPR at 5% FPR saturates at 1.0 on 34 tuning-window frauds
    )


def make_synthetic_baf(n: int = 8000, seed: int = 0) -> pd.DataFrame:
    """Small frame with the BAF schema for tests and CI. Fraud rate drifts upward over months."""
    rng = np.random.default_rng(seed)
    month = np.sort(rng.integers(0, 8, size=n))
    age = rng.choice([20, 30, 40, 50, 60, 70], size=n, p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05])
    df = pd.DataFrame(
        {
            "income": rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), size=n),
            "name_email_similarity": rng.beta(2, 2, size=n),
            "prev_address_months_count": np.where(rng.random(n) < 0.7, -1, rng.integers(0, 300, n)),
            "current_address_months_count": rng.integers(0, 300, size=n),
            BAF_AGE: age,
            "days_since_request": rng.exponential(1.0, size=n),
            "intended_balcon_amount": np.where(rng.random(n) < 0.7, -1.0, rng.exponential(20, n)),
            "payment_type": rng.choice(BAF_CATEGORICAL["payment_type"], size=n),
            "zip_count_4w": rng.integers(1, 3000, size=n),
            "velocity_6h": rng.normal(5000, 1500, size=n),
            "velocity_24h": rng.normal(4800, 1000, size=n),
            "velocity_4w": rng.normal(4700, 700, size=n),
            "bank_branch_count_8w": rng.integers(0, 500, size=n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 20, size=n),
            "employment_status": rng.choice(BAF_CATEGORICAL["employment_status"], size=n),
            "credit_risk_score": rng.normal(130, 50, size=n).astype(int),
            "email_is_free": rng.integers(0, 2, size=n),
            "housing_status": rng.choice(BAF_CATEGORICAL["housing_status"], size=n),
            "phone_home_valid": rng.integers(0, 2, size=n),
            "phone_mobile_valid": rng.integers(0, 2, size=n),
            "bank_months_count": np.where(rng.random(n) < 0.25, -1, rng.integers(0, 30, n)),
            "has_other_cards": rng.integers(0, 2, size=n),
            "proposed_credit_limit": rng.choice([200.0, 500.0, 1000.0, 2000.0], size=n),
            "foreign_request": (rng.random(n) < 0.02).astype(int),
            "source": rng.choice(BAF_CATEGORICAL["source"], size=n, p=[0.99, 0.01]),
            "session_length_in_minutes": rng.exponential(6.0, size=n),
            "device_os": rng.choice(BAF_CATEGORICAL["device_os"], size=n),
            "keep_alive_session": rng.integers(0, 2, size=n),
            "device_distinct_emails_8w": rng.integers(0, 3, size=n),
            "device_fraud_count": np.zeros(n, dtype=int),
            BAF_MONTH: month,
        }
    )
    logit = (
        -5.0
        + 0.15 * month
        + 2.0 * (df["name_email_similarity"] < 0.2)
        + 0.004 * (df["credit_risk_score"] - 130)
        + 1.2 * (df["device_os"] == "windows")
        + 0.8 * (df["proposed_credit_limit"] >= 1000)
        + 0.7 * (age >= 50)
    )
    df.insert(0, BAF_TARGET, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return df

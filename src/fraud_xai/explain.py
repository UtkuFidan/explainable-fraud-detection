"""Per-case reason codes from TreeSHAP contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def reason_codes(
    shap_values: np.ndarray, X: pd.DataFrame, reference: pd.DataFrame, top: int = 3
) -> list[list[dict]]:
    """Top-``top`` features per row by |SHAP|, with direction and where the value sits.

    For numeric features ``percentile`` is the share of reference (training) rows with a lower
    value, so a reviewer can see how unusual it is. Categorical features report their level and
    the share of training rows that have it.
    """
    cols = list(X.columns)
    cat = {c for c in cols if isinstance(X[c].dtype, pd.CategoricalDtype)}
    ref_sorted = {c: np.sort(reference[c].dropna().to_numpy(float)) for c in cols if c not in cat}
    ref_share = {c: reference[c].astype(str).value_counts(normalize=True) for c in cat}
    out = []
    for i in range(len(X)):
        codes = []
        for j in np.argsort(-np.abs(shap_values[i]))[:top]:
            c, raw = cols[j], X.iloc[i, j]
            s = float(shap_values[i, j])
            if c in cat:
                value, pct = str(raw), float(ref_share[c].get(str(raw), 0.0))
            elif pd.isna(raw):
                value, pct = None, None
            else:
                value = float(raw)
                pct = float(np.searchsorted(ref_sorted[c], value, side="left") / len(ref_sorted[c]))
            codes.append(
                {
                    "feature": c,
                    "value": value,
                    "percentile": pct,
                    "categorical": c in cat,
                    "shap": s,
                    "direction": "increases" if s > 0 else "decreases",
                }
            )
        out.append(codes)
    return out

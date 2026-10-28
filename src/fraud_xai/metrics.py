"""Metrics for imbalanced fraud screening: BAF's TPR at fixed FPR, PR-AUC, review-budget recall.

Scores are margins (log-odds) so that models and blends share one scale. The bootstrap resamples
rows by integer weights over a score-sorted array, which avoids re-sorting on every replicate.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve

FPR_TARGET = 0.05


def tpr_at_fpr(y_true, score, fpr: float = FPR_TARGET) -> float:
    """BAF's headline metric: best recall among operating points with FPR <= ``fpr``."""
    f, t, _ = roc_curve(y_true, score, drop_intermediate=False)
    return float(t[f <= fpr].max())


def pr_auc(y_true, score) -> float:
    return float(average_precision_score(y_true, score))


def recall_at_frac(y_true, score, frac: float) -> float:
    """Share of all fraud caught when reviewers work the top ``frac`` of scored cases."""
    y = np.asarray(y_true)
    k = max(1, int(np.ceil(frac * len(y))))
    if y.sum() == 0:
        return 0.0
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def threshold_at_fpr(y_true, score, fpr: float = FPR_TARGET) -> float:
    """Score threshold that lets through at most ``fpr`` of the legitimate cases."""
    s = np.sort(np.asarray(score)[np.asarray(y_true) == 0])
    return float(s[min(int(np.ceil((1 - fpr) * len(s))), len(s) - 1)])


# ---- bootstrap ---------------------------------------------------------------------------------
def _sorted_curves(y, s):
    order = np.argsort(-np.asarray(s), kind="stable")
    return order, np.asarray(y)[order].astype(float)


def _w_tpr(ys, w, fpr):
    tp, fp = np.cumsum(w * ys), np.cumsum(w * (1 - ys))
    ok = fp <= fpr * fp[-1]
    return float(tp[ok].max() / tp[-1]) if ok.any() else 0.0


def _w_pr_auc(ys, w):
    tp, fp = np.cumsum(w * ys), np.cumsum(w * (1 - ys))
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float((w * ys * prec).sum() / tp[-1])


def _w_recall(ys, w, frac):
    cw = np.cumsum(w)
    i = int(np.searchsorted(cw, np.ceil(frac * cw[-1]), side="left"))
    return float(np.cumsum(w * ys)[min(i, len(ys) - 1)] / (w * ys).sum())


def bootstrap_metrics(y, s, budget_frac: float, n_boot: int = 500, seed: int = 0) -> dict:
    """Point estimates plus 95% percentile intervals for the three headline metrics."""
    order, ys = _sorted_curves(y, s)
    n = len(ys)
    rng = np.random.default_rng(seed)
    reps = {"tpr_at_5fpr": [], "pr_auc": [], "recall_at_budget": []}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        reps["tpr_at_5fpr"].append(_w_tpr(ys, w, FPR_TARGET))
        reps["pr_auc"].append(_w_pr_auc(ys, w))
        reps["recall_at_budget"].append(_w_recall(ys, w, budget_frac))
    point = {
        "tpr_at_5fpr": tpr_at_fpr(y, s),
        "pr_auc": pr_auc(y, s),
        "recall_at_budget": recall_at_frac(y, s, budget_frac),
    }
    return {
        k: {
            "value": point[k],
            "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)),
        }
        for k, v in reps.items()
    }


# ---- calibration -------------------------------------------------------------------------------
def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE with equal-mass bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.array_split(np.argsort(p, kind="stable"), bins)
    return float(sum(len(i) / len(y) * abs(y[i].mean() - p[i].mean()) for i in edges))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


class Calibrator:
    """Maps a margin to a probability: Platt (sigmoid) or isotonic, fitted on the cal window."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, margin, y):
        m = np.asarray(margin, float)
        if self.kind == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(m.reshape(-1, 1), y)
        else:
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(m, y)
        return self

    def predict(self, margin):
        m = np.asarray(margin, float)
        if self.kind == "platt":
            return self.model_.predict_proba(m.reshape(-1, 1))[:, 1]
        return self.model_.predict(m)


def pick_calibrator(margin, y, seed: int = 0) -> tuple[Calibrator, dict]:
    """Choose Platt or isotonic by 2-fold cross-fitted Brier score, then refit on everything."""
    m, y = np.asarray(margin, float), np.asarray(y)
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, 2)
    scores = {}
    for kind in ("platt", "isotonic"):
        pred = np.empty(len(y))
        for k in range(2):
            fit, hold = folds[1 - k], folds[k]
            pred[hold] = Calibrator(kind).fit(m[fit], y[fit]).predict(m[hold])
        scores[kind] = brier(y, pred)
    best = min(scores, key=scores.get)
    return Calibrator(best).fit(m, y), scores


# ---- fairness: predictive equality -------------------------------------------------------------
def group_rates(y, score, older, thr_young: float, thr_old: float) -> dict:
    """FPR / TPR / alert rate by age group when each group has its own threshold."""
    y, older = np.asarray(y).astype(bool), np.asarray(older).astype(bool)
    alert = np.where(older, np.asarray(score) >= thr_old, np.asarray(score) >= thr_young)
    out = {}
    for name, g in (("under_50", ~older), ("over_50", older)):
        neg, pos = g & ~y, g & y
        out[name] = {
            "fpr": float(alert[neg].mean()),
            "tpr": float(alert[pos].mean()),
            "alert_rate": float(alert[g].mean()),
            "n": int(g.sum()),
            "fraud": int(pos.sum()),
        }
    neg, pos = ~y, y
    out["overall"] = {"fpr": float(alert[neg].mean()), "tpr": float(alert[pos].mean())}
    out["fpr_ratio_over_under"] = out["over_50"]["fpr"] / max(out["under_50"]["fpr"], 1e-12)
    out["fpr_gap"] = out["over_50"]["fpr"] - out["under_50"]["fpr"]
    return out


def bootstrap_group_gap(
    y, score, older, thr_young, thr_old, n_boot: int = 500, seed: int = 0
) -> dict:
    """95% intervals for the FPR ratio and gap (over-50 vs under-50) and overall TPR."""
    y, older = np.asarray(y).astype(bool), np.asarray(older).astype(bool)
    alert = np.where(older, np.asarray(score) >= thr_old, np.asarray(score) >= thr_young)
    n = len(y)
    rng = np.random.default_rng(seed)
    ratios, gaps, tprs = [], [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        fo = (w * alert * older * ~y).sum() / (w * older * ~y).sum()
        fy = (w * alert * ~older * ~y).sum() / (w * ~older * ~y).sum()
        ratios.append(fo / max(fy, 1e-12))
        gaps.append(fo - fy)
        tprs.append((w * alert * y).sum() / (w * y).sum())
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"fpr_ratio_ci": q(ratios), "fpr_gap_ci": q(gaps), "overall_tpr_ci": q(tprs)}


# ---- drift -------------------------------------------------------------------------------------
def psi(reference, current, bins: int = 10) -> float:
    """Population stability index of one numeric column against a reference window."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))

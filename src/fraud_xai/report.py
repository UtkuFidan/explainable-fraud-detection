"""Figures and the reviewer case file, built from the bundle written by ``train``."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .explain import reason_codes  # noqa: E402
from .pipeline import NAMES  # noqa: E402

# Okabe-Ito colours: distinguishable under common colour-vision deficiencies
COLORS = {
    "logistic_regression": "#7f7f7f",
    "lightgbm": "#0072B2",
    "xgboost": "#D55E00",
    "catboost": "#009E73",
    "blend": "#CC79A7",
}
INK = "#222222"
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _models(test: pd.DataFrame) -> list[str]:
    return [k for k in COLORS if f"margin_{k}" in test]


def _pr_plot(test, deployed, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k in _models(test):
        p, r, _ = precision_recall_curve(test["y"], test[f"margin_{k}"])
        ax.plot(r, p, color=COLORS[k], label=NAMES[k], lw=2 if k == deployed else 1.1)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision/recall, held-out test window")
    ax.legend(fontsize=7.5, frameon=False)
    _save(fig, path)


def _calibration_plot(test, metrics, path):
    """Reliability on equal-mass bins for the raw booster output and the chosen calibrator."""
    y = test["y"].to_numpy()
    raw = 1 / (1 + np.exp(-test[f"margin_{metrics['deployed']}"].to_numpy()))
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.plot([0, 0.6], [0, 0.6], color="#888888", lw=1, ls="--", label="perfect")
    kind = metrics["calibration"]["chosen"]
    for p, label, c in ((raw, "raw output", "#D55E00"), (test["prob"].to_numpy(), kind, "#0072B2")):
        bins = np.array_split(np.argsort(p), 15)
        ax.plot(
            [p[b].mean() for b in bins],
            [y[b].mean() for b in bins],
            "o-",
            ms=4,
            lw=1,
            color=c,
            label=label,
        )
    ax.set(
        xlabel="Predicted probability",
        ylabel="Observed fraud rate",
        title="Calibration, test window (15 equal-mass bins)",
        xlim=(0, 0.6),
        ylim=(0, 0.6),
    )
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def _drift_plot(metrics, path):
    months = sorted(metrics["by_month"], key=int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    for k in COLORS:
        if k not in metrics["by_month"][months[0]]:
            continue
        a.plot(
            [int(m) for m in months],
            [metrics["by_month"][m][k]["tpr_at_5fpr"] for m in months],
            "o-",
            color=COLORS[k],
            label=NAMES[k],
            lw=1.2,
        )
    a.set(
        xlabel="Test month",
        ylabel="TPR at 5% FPR",
        title="Performance by test month",
        xticks=[int(m) for m in months],
    )
    a.legend(fontsize=7, frameon=False)
    psi = metrics["psi_train_vs_test"][::-1]
    b.barh([r["feature"] for r in psi], [r["psi"] for r in psi], color="#0072B2", height=0.6)
    b.axvline(0.1, color="#888888", lw=0.8, ls="--")
    b.set(xlabel="PSI, train months vs test months (dashed: 0.1)", title="Largest feature shifts")
    b.grid(axis="y", visible=False)
    _save(fig, path)


def _fairness_plot(metrics, path):
    f = metrics["fairness"]["deployed"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(2)
    for i, (key, label, c) in enumerate(
        (
            ("global_threshold", "one global threshold", "#D55E00"),
            ("group_thresholds", "threshold per age group", "#0072B2"),
        )
    ):
        vals = [f[key]["under_50"]["fpr"], f[key]["over_50"]["fpr"]]
        bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=c, label=label)
        for r, v in zip(bars, vals, strict=True):
            ax.text(
                r.get_x() + r.get_width() / 2,
                v + 0.002,
                f"{v:.1%}",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.axhline(0.05, color="#888888", lw=0.8, ls="--")
    ax.set(
        xticks=x,
        xticklabels=["under 50", "50 and over"],
        ylabel="False positive rate",
        title="FPR by age group, test months 6-7",
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def _fmt(c: dict) -> str:
    arrow = "raises" if c["direction"] == "increases" else "lowers"
    if c["value"] is None:
        where = "missing"
    elif c["categorical"]:
        where = f"= {c['value']} ({c['percentile'] * 100:.0f}% of training rows)"
    else:
        where = f"= {c['value']:.4g} (training percentile {c['percentile'] * 100:.0f})"
    return f"`{c['feature']}` {where} {arrow} the score ({c['shap']:+.2f} log-odds)"


CHECKLIST = [
    "Verify the applicant's identity documents against the declared date of birth and address",
    "Check the email, phone and device against other recent applications "
    "(shared device, repeated email patterns)",
    "Check the features named above for a legitimate explanation",
    "Contact the applicant where policy requires it",
    "Record the outcome: confirmed fraud, false positive or escalate, with a one-line reason",
]


def build_report(
    artifacts: str | Path,
    docs_dir: str | Path,
    reports_dir: str | Path,
    top_n: int = 15,
    max_global: int = 4000,
) -> Path:
    b = joblib.load(Path(artifacts) / "bundle.joblib")
    docs, reports = Path(docs_dir), Path(reports_dir)
    docs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    m, model, ref, thr = b["metrics"], b["model"], b["reference"], b["threshold"]
    test = b["test"].reset_index(drop=True)
    feats = list(ref.columns)
    deployed = b["deployed"]

    _pr_plot(test, deployed, docs / "pr_curves.png")
    _calibration_plot(test, m, docs / "calibration.png")
    if "by_month" in m:
        _drift_plot(m, docs / "drift.png")
    if "fairness" in m:
        _fairness_plot(m, docs / "fairness.png")

    # global SHAP on every test fraud plus a random sample of the rest
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(test["y"] == 1)
    neg_all = np.flatnonzero(test["y"] == 0)
    neg = rng.choice(neg_all, size=min(max_global, len(neg_all)), replace=False)
    sample = test.loc[np.concatenate([pos, neg]), feats]
    sv = model.shap(sample)
    plot_X = sample.apply(lambda s: s.cat.codes if isinstance(s.dtype, pd.CategoricalDtype) else s)
    plt.figure()
    shap.summary_plot(sv, plot_X, max_display=12, show=False, plot_size=(7.5, 5))
    plt.title("SHAP summary, test window (log-odds of fraud)", fontsize=10)
    plt.tight_layout()
    plt.savefig(docs / "shap_summary.png", dpi=110)
    plt.close("all")

    flagged = test[test[f"margin_{deployed}"] >= thr].sort_values(
        f"margin_{deployed}", ascending=False
    )
    n_flagged, n_caught = len(flagged), int(flagged["y"].sum())
    flagged = flagged.head(top_n)
    codes = reason_codes(model.shap(flagged[feats]), flagged[feats], ref, top=3)

    d = m["dataset"]["test"]
    lines = [
        "# Reviewer case file: flagged applications",
        "",
        f"Model: `{deployed}` with TreeSHAP reason codes. Test window: months 6 and 7, "
        f"{d['rows']:,} applications, {d['fraud']:,} confirmed fraud.",
        f"The alert threshold is the score that lets through 5% of legitimate applications on "
        f"the calibration window (month 5). On the test window it produced {n_flagged:,} "
        f"alerts, of which {n_caught:,} are fraud ({n_caught / max(n_flagged, 1):.1%} precision). "
        f"The {len(flagged)} highest-scoring are listed below.",
        "",
        "`Label` is shown only so the examples can be checked; a reviewer would not have it.",
        "",
    ]
    for rank, (r, cs) in enumerate(zip(flagged.to_dict("records"), codes, strict=True), 1):
        lines += [
            f"## Case {rank}: calibrated probability {r['prob']:.2f}",
            "",
            f"- Month {int(r['month'])}, customer age {int(r['customer_age'])}",
            f"- Label (demo only): {'FRAUD' if r['y'] == 1 else 'legitimate'}",
            "- Why it was flagged:",
            *[f"  {i}. {_fmt(c)}" for i, c in enumerate(cs, 1)],
            "- Review checklist:",
            *[f"  - [ ] {item}" for item in CHECKLIST],
            "",
        ]
    path = reports / "flagged_cases.md"
    path.write_text("\n".join(lines))
    return path

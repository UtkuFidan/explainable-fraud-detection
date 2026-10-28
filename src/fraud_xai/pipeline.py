"""Experiment driver: tune, fit, blend, calibrate, evaluate (with drift and fairness), save.

Windows (see ``data.Task``): train fits the models; ``tune`` drives early stopping and the Optuna
objective; ``cal`` is untouched by tuning and is used for calibration, model selection and alert
thresholds; ``test`` is used once, for the reported numbers.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from . import metrics as M
from .data import BAF_AGE, Task, baf_task, load_baf
from .models import FAMILIES, Booster, fit_logistic, tune

NAMES = {
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "blend": "Blend (mean of 3 boosters)",
}


@dataclass
class Config:
    trials: dict = field(default_factory=lambda: {"lightgbm": 60, "xgboost": 50, "catboost": 30})
    timeout: dict = field(  # seconds per family; the trial that is running at the limit finishes
        default_factory=lambda: {"lightgbm": 720.0, "xgboost": 600.0, "catboost": 1200.0}
    )
    tune_rows: dict = field(  # training rows used during the search, per family
        default_factory=lambda: {"lightgbm": 400_000, "xgboost": 100_000, "catboost": 60_000}
    )
    seed: int = 0
    n_jobs: int = -1
    n_boot: int = 500
    reuse_params: str | None = None  # path to a best_params.json from an earlier run


def _logger(path: Path):
    def log(msg: str) -> None:
        print(msg, flush=True)
        with path.open("a") as f:
            f.write(msg + "\n")

    return log


def _fit_final(family, params, task: Task, cfg: Config, drop: list[str] | None = None) -> Booster:
    drop = drop or []
    X_tr, y_tr = task.part("train")
    X_tu, y_tu = task.part("tune")
    cats = [c for c in task.cat_cols if c not in drop]
    return Booster(family, params, cats, cfg.seed, cfg.n_jobs).fit(
        X_tr.drop(columns=drop), y_tr, X_tu.drop(columns=drop), y_tu
    )


def _fairness(task: Task, s_cal, s_te, cfg: Config, label: str) -> dict:
    """Predictive equality across age groups: one global threshold vs one threshold per group."""
    y_cal, y_te = task.y[task.cal], task.y[task.test]
    o_cal, o_te = task.older[task.cal], task.older[task.test]
    thr = M.threshold_at_fpr(y_cal, s_cal)
    t_young = M.threshold_at_fpr(y_cal[~o_cal], s_cal[~o_cal])
    t_old = M.threshold_at_fpr(y_cal[o_cal], s_cal[o_cal])
    out = {"model": label, "thresholds": {"global": thr, "under_50": t_young, "over_50": t_old}}
    setups = {"global_threshold": (thr, thr), "group_thresholds": (t_young, t_old)}
    for name, (a, b) in setups.items():
        out[name] = M.group_rates(y_te, s_te, o_te, a, b) | M.bootstrap_group_gap(
            y_te, s_te, o_te, a, b, cfg.n_boot, cfg.seed
        )
    return out


def _by_month(task: Task, scores: dict, thr: float, deployed: str) -> dict:
    te = task.test
    month, y = task.period[te], task.y[te]
    out = {}
    for m in sorted(set(month)):
        k = month == m
        row = {"rows": int(k.sum()), "fraud": int(y[k].sum()), "fraud_rate": float(y[k].mean())}
        for name, s in scores.items():
            row[name] = {
                "tpr_at_5fpr": M.tpr_at_fpr(y[k], s[k]),
                "pr_auc": M.pr_auc(y[k], s[k]),
            }
        sd = scores[deployed][k]
        row["fixed_threshold"] = {  # threshold set on the calibration window (month 5)
            "alert_rate": float((sd >= thr).mean()),
            "fpr": float((sd[y[k] == 0] >= thr).mean()),
            "tpr": float((sd[y[k] == 1] >= thr).mean()),
        }
        out[int(m)] = row
    return out


def _psi_table(task: Task, top: int = 8) -> list[dict]:
    num = [c for c in task.X.columns if c not in task.cat_cols]
    tr, te = task.X.iloc[task.train], task.X.iloc[task.test]
    rows = [{"feature": c, "psi": M.psi(tr[c], te[c])} for c in num]
    return sorted(rows, key=lambda r: -r["psi"])[:top]


def random_vs_temporal(task: Task, family: str, params: dict, cfg: Config) -> dict:
    """Same model and window sizes, but rows drawn at random instead of in time order."""
    n = len(task.y)
    perm = np.random.default_rng(cfg.seed).permutation(n)
    a, b, c = len(task.train), len(task.tune), len(task.test)
    rnd = Task(
        name=task.name + " (random split)",
        X=task.X,
        y=task.y,
        train=np.sort(perm[:a]),
        tune=np.sort(perm[a : a + b]),
        cal=np.sort(perm[a + b : a + b + len(task.cal)]),
        test=np.sort(perm[n - c :]),
        cat_cols=task.cat_cols,
    )
    res = {}
    for label, t in (("temporal", task), ("random", rnd)):
        m = _fit_final(family, params, t, cfg)
        X_te, y_te = t.part("test")
        s = m.margin(X_te)
        res[label] = {"tpr_at_5fpr": M.tpr_at_fpr(y_te, s), "pr_auc": M.pr_auc(y_te, s)}
    return res


def run_experiment(task: Task, out_dir: str | Path, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out / "train.log")
    t0 = time.time()
    y_cal, y_te = task.y[task.cal], task.y[task.test]
    X_cal, X_te = task.X.iloc[task.cal], task.X.iloc[task.test]

    # 1. hyperparameters
    tuned = json.loads(Path(cfg.reuse_params).read_text()) if cfg.reuse_params else {}
    for fam in FAMILIES:
        if fam in tuned:
            continue
        r = tune(
            fam,
            task,
            cfg.trials[fam],
            cfg.timeout[fam],
            cfg.tune_rows[fam],
            cfg.seed,
            cfg.n_jobs,
            log,
        )
        tuned[fam] = {
            "params": r.params,
            "tune_metric": task.tune_metric,
            "tune_value": r.best_value,
            "trials": r.n_trials,
            "seconds": r.seconds,
            "tune_rows": min(cfg.tune_rows[fam], len(task.train)),
        }
        (out / "best_params.json").write_text(json.dumps(tuned, indent=2))

    # 2. final fits on the whole training window
    fitted: dict = {}
    cal_s, te_s, info = {}, {}, {}
    lr = fit_logistic(task, cfg.seed)
    cal_s["logistic_regression"] = lr.decision_function(X_cal)
    te_s["logistic_regression"] = lr.decision_function(X_te)
    for fam in FAMILIES:
        t1 = time.time()
        fitted[fam] = _fit_final(fam, tuned[fam]["params"], task, cfg)
        cal_s[fam], te_s[fam] = fitted[fam].margin(X_cal), fitted[fam].margin(X_te)
        info[fam] = {"trees": fitted[fam].n_trees, "fit_seconds": round(time.time() - t1, 1)}
        log(f"[{fam}] final fit {info[fam]}")
    cal_s["blend"] = np.mean([cal_s[f] for f in FAMILIES], axis=0)
    te_s["blend"] = np.mean([te_s[f] for f in FAMILIES], axis=0)

    # 3. selection on the calibration window
    cal_tpr = {k: M.tpr_at_fpr(y_cal, s) for k, s in cal_s.items()}
    deployed = max(FAMILIES, key=lambda f: cal_tpr[f])
    selected = max(cal_tpr, key=cal_tpr.get)
    log(f"cal TPR@5%FPR {cal_tpr}; deployed={deployed}; best on cal={selected}")

    # 4. test metrics with bootstrap intervals
    table = {
        k: M.bootstrap_metrics(y_te, s, task.budget_frac, cfg.n_boot, cfg.seed)
        for k, s in te_s.items()
    }

    # 5. calibration of the deployed model
    cal_obj, cal_brier = M.pick_calibrator(cal_s[deployed], y_cal, cfg.seed)
    p_te = cal_obj.predict(te_s[deployed])
    platt = M.Calibrator("platt").fit(cal_s[deployed], y_cal).predict(te_s[deployed])
    iso = M.Calibrator("isotonic").fit(cal_s[deployed], y_cal).predict(te_s[deployed])
    calibration = {
        "chosen": cal_obj.kind,
        "cross_fit_brier_on_cal": cal_brier,
        **{
            name: {"brier": M.brier(y_te, p), "ece": M.expected_calibration_error(y_te, p)}
            for name, p in (
                ("raw", expit(te_s[deployed])),
                ("platt", platt),
                ("isotonic", iso),
            )
        },
    }

    thr = M.threshold_at_fpr(y_cal, cal_s[deployed])
    metrics: dict = {
        "task": task.name,
        "dataset": {
            w: {"rows": len(getattr(task, w)), "fraud": int(task.y[getattr(task, w)].sum())}
            for w in ("train", "tune", "cal", "test")
        },
        "budget_frac": task.budget_frac,
        "tuning": tuned,
        "final_models": info,
        "cal_tpr_at_5fpr": cal_tpr,
        "deployed": deployed,
        "best_on_cal": selected,
        "test": table,
        "calibration": calibration,
        "alert_threshold_margin": thr,
        "psi_train_vs_test": _psi_table(task),
        "config": asdict(cfg),
    }

    # 6. drift and fairness (BAF only)
    if task.period is not None:
        metrics["by_month"] = _by_month(task, te_s, thr, deployed)
    if task.older is not None:
        fair = {"deployed": _fairness(task, cal_s[deployed], te_s[deployed], cfg, deployed)}
        drop = [BAF_AGE, "address_share_of_life"]
        nb = _fit_final(deployed, tuned[deployed]["params"], task, cfg, drop)
        keep = [c for c in task.X.columns if c not in drop]
        s_cal_na, s_te_na = nb.margin(X_cal[keep]), nb.margin(X_te[keep])
        fair["without_age"] = _fairness(task, s_cal_na, s_te_na, cfg, deployed + " without age")
        fair["without_age"]["test_tpr_at_5fpr"] = M.tpr_at_fpr(y_te, s_te_na)
        old, tr, te = task.older, task.train, task.test
        fair["prevalence"] = {
            "train_over_50": float(task.y[tr][old[tr]].mean()),
            "train_under_50": float(task.y[tr][~old[tr]].mean()),
            "test_over_50": float(task.y[te][old[te]].mean()),
            "test_under_50": float(task.y[te][~old[te]].mean()),
            "share_over_50_test": float(old[te].mean()),
        }
        metrics["fairness"] = fair
    metrics["random_vs_temporal"] = random_vs_temporal(
        task, deployed, tuned[deployed]["params"], cfg
    )
    metrics["seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rng = np.random.default_rng(cfg.seed)
    ref = task.X.iloc[task.train].iloc[
        rng.choice(len(task.train), min(20_000, len(task.train)), replace=False)
    ]
    extra = {"y": y_te, "prob": p_te} | {f"margin_{k}": v for k, v in te_s.items()}
    if task.period is not None:
        extra["month"] = task.period[task.test]
    if task.older is not None:
        extra["older"] = task.older[task.test]
    joblib.dump(
        {
            "model": fitted[deployed],
            "deployed": deployed,
            "calibrator": cal_obj,
            "threshold": thr,
            "reference": ref,
            "test": X_te.assign(**extra),
            "metrics": metrics,
        },
        out / "bundle.joblib",
    )
    return metrics


def variant_sweep(data_dir: str, params: dict, variants: list[str], cfg: Config) -> dict:
    """Retrain one fixed LightGBM configuration on every BAF variant (temporal split)."""
    res = {}
    for v in variants:
        task = baf_task(load_baf(data_dir, v), v, cfg.seed)
        m = _fit_final("lightgbm", params, task, cfg)
        s_cal = m.margin(task.X.iloc[task.cal])
        s_te = m.margin(task.X.iloc[task.test])
        y_te = task.y[task.test]
        fair = _fairness(task, s_cal, s_te, cfg, v)
        old = task.older

        def brief(f):
            return {
                "fpr_under_50": f["under_50"]["fpr"],
                "fpr_over_50": f["over_50"]["fpr"],
                "fpr_ratio": f["fpr_ratio_over_under"],
                "overall_tpr": f["overall"]["tpr"],
            }

        res[v] = {
            "rows": len(task.y),
            "fraud_rate": float(task.y.mean()),
            "share_over_50": float(old.mean()),
            "fraud_rate_over_50": float(task.y[old].mean()),
            "fraud_rate_under_50": float(task.y[~old].mean()),
            "tpr_at_5fpr": M.tpr_at_fpr(y_te, s_te),
            "pr_auc": M.pr_auc(y_te, s_te),
            "global_threshold": brief(fair["global_threshold"]),
            "group_thresholds": brief(fair["group_thresholds"]),
        }
    return res

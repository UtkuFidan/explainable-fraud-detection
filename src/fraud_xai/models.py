"""Model families, search spaces and the Optuna tuning loop.

Every model exposes ``margin(X)`` (log-odds) and, for the tree models, ``shap(X)`` (native
TreeSHAP contributions). Categoricals are handled natively by the three boosters and one-hot
encoded for logistic regression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from xgboost import DMatrix, XGBClassifier

from .data import Task
from .metrics import pr_auc, tpr_at_fpr

FAMILIES = ("lightgbm", "xgboost", "catboost")
MAX_TREES = 2000
STOP_ROUNDS = 100


class Booster:
    """Uniform wrapper over LightGBM, XGBoost and CatBoost classifiers."""

    def __init__(self, family: str, params: dict, cat_cols: list[str], seed: int, n_jobs: int):
        self.family, self.cat_cols = family, cat_cols
        p = dict(params)
        if family == "lightgbm":
            self.model = LGBMClassifier(
                n_estimators=MAX_TREES, random_state=seed, n_jobs=n_jobs, verbose=-1, **p
            )
        elif family == "xgboost":
            self.model = XGBClassifier(
                n_estimators=MAX_TREES,
                tree_method="hist",
                enable_categorical=True,
                eval_metric="auc",
                early_stopping_rounds=STOP_ROUNDS,
                random_state=seed,
                n_jobs=n_jobs,
                **p,
            )
        else:
            self.model = CatBoostClassifier(
                iterations=MAX_TREES,
                eval_metric="AUC",
                od_type="Iter",
                od_wait=STOP_ROUNDS,
                random_seed=seed,
                thread_count=n_jobs,
                verbose=False,
                allow_writing_files=False,
                **p,
            )

    def _cb(self, X: pd.DataFrame, y=None) -> Pool:
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(str)
        return Pool(X, label=y, cat_features=self.cat_cols)

    def fit(self, X, y, X_es, y_es) -> Booster:
        if self.family == "lightgbm":
            self.model.fit(
                X,
                y,
                eval_set=[(X_es, y_es)],
                eval_metric="auc",
                callbacks=[early_stopping(STOP_ROUNDS, verbose=False)],
            )
        elif self.family == "xgboost":
            self.model.fit(X, y, eval_set=[(X_es, y_es)], verbose=False)
        else:
            self.model.fit(self._cb(X, y), eval_set=self._cb(X_es, y_es), use_best_model=True)
        return self

    @property
    def n_trees(self) -> int:
        if self.family == "lightgbm":
            return int(self.model.best_iteration_)
        if self.family == "xgboost":
            return int(self.model.best_iteration) + 1
        return int(self.model.get_best_iteration()) + 1

    def margin(self, X) -> np.ndarray:
        if self.family == "lightgbm":
            return self.model.predict(X, raw_score=True)
        if self.family == "xgboost":
            return self.model.predict(X, output_margin=True)
        return self.model.predict(self._cb(X), prediction_type="RawFormulaVal")

    def shap(self, X) -> np.ndarray:
        """TreeSHAP contributions in log-odds, shape (rows, features); bias term dropped."""
        if self.family == "lightgbm":
            return self.model.predict(X, pred_contrib=True)[:, :-1]
        if self.family == "xgboost":
            dm = DMatrix(X, enable_categorical=True)
            return self.model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
        return self.model.get_feature_importance(self._cb(X), type="ShapValues")[:, :-1]


def suggest(family: str, trial: optuna.Trial) -> dict:
    if family == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True),
        }
    if family == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True),
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.06, 0.25, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 3),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 50, log=True),
    }


@dataclass
class TuneResult:
    params: dict
    best_value: float
    n_trials: int
    seconds: float


def tune(
    family: str,
    task: Task,
    n_trials: int,
    timeout: float,
    tune_rows: int,
    seed: int,
    n_jobs: int,
    log=print,
) -> TuneResult:
    """TPE search. Fit on a row sample of the training window, early-stop and score on ``tune``.

    The objective is ``task.tune_metric`` on the tuning window (TPR at 5% FPR for BAF).
    The calibration window and the test window are never touched here.
    """
    X_tr, y_tr = task.part("train")
    if len(X_tr) > tune_rows:
        keep = np.sort(np.random.default_rng(seed).choice(len(X_tr), tune_rows, replace=False))
        X_tr, y_tr = X_tr.iloc[keep], y_tr[keep]
    X_tu, y_tu = task.part("tune")

    def objective(trial: optuna.Trial) -> float:
        b = Booster(family, suggest(family, trial), task.cat_cols, seed, n_jobs)
        b.fit(X_tr, y_tr, X_tu, y_tu)
        trial.set_user_attr("trees", b.n_trees)
        score = b.margin(X_tu)
        return pr_auc(y_tu, score) if task.tune_metric == "pr_auc" else tpr_at_fpr(y_tu, score)

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        log(
            f"[{family}] trial {trial.number:>2} value={trial.value:.4f} "
            f"best={study.best_value:.4f} trees={trial.user_attrs.get('trees')}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[progress])
    return TuneResult(
        params=study.best_params | ({"subsample_freq": 1} if family == "lightgbm" else {}),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        seconds=round(time.time() - t0, 1),
    )


def fit_logistic(task: Task, seed: int = 0) -> Pipeline:
    """Regularised logistic regression on quantile-normalised numerics and one-hot categoricals.

    C is picked on the tuning window from a short grid.
    """
    num = [c for c in task.X.columns if c not in task.cat_cols]
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True),
                    QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                ),
                num,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), task.cat_cols),
        ]
    )
    X_tr, y_tr = task.part("train")
    X_tu, y_tu = task.part("tune")
    best, best_val = None, -1.0
    for C in (0.01, 0.1, 1.0):
        pipe = make_pipeline(
            pre, LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed)
        ).fit(X_tr, y_tr)
        val = tpr_at_fpr(y_tu, pipe.decision_function(X_tu))
        if val > best_val:
            best, best_val = pipe, val
    return best

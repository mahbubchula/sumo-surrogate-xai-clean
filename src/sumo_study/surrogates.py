"""Surrogate model zoo, development cross-validation, frozen fitting and prediction.

All architectures and hyperparameters are fixed in advance (no tuning on calibration or test
data). The prespecified primary surrogate is the equal-weight CatBoost + XGBoost ensemble.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

from .config import FEATURES

ENSEMBLE = ("CatBoost", "XGBoost")
MODEL_ORDER = ["Ridge", "GaussianProcess", "RandomForest", "LightGBM", "CatBoost", "XGBoost",
               "Ensemble"]


def model_zoo(size: int = 500) -> dict:
    kernel = (ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=np.ones(len(FEATURES)),
                                                        length_scale_bounds=(1e-2, 1e2), nu=2.5)
              + WhiteKernel(0.1, (1e-5, 1e1)))
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "GaussianProcess": make_pipeline(MinMaxScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42)),
        "RandomForest": RandomForestRegressor(n_estimators=size, max_depth=8, random_state=42,
                                              n_jobs=1),
        "LightGBM": LGBMRegressor(n_estimators=size, learning_rate=.05, num_leaves=15,
                                  min_child_samples=5, verbosity=-1, random_state=42, n_jobs=1),
        "CatBoost": CatBoostRegressor(iterations=size, learning_rate=.05, depth=5, random_seed=42,
                                      thread_count=1, verbose=False, allow_writing_files=False),
        "XGBoost": XGBRegressor(n_estimators=size, learning_rate=.05, max_depth=4, subsample=.85,
                                colsample_bytree=.85, random_state=42, n_jobs=1),
    }


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(R2=float(r2_score(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), bias=float(np.mean(p - y)),
                MAPE_pct=float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-6))) * 100))


def _fit(model, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clone(model).fit(X, y)


class Surrogate:
    """Frozen per-target models. ``predict(X, target, name)``; name='Ensemble' is primary."""

    def __init__(self, fitted: dict):
        self.fitted = fitted

    def predict(self, X, target: str, name: str = "Ensemble") -> np.ndarray:
        X = pd.DataFrame(X, columns=FEATURES)
        if name == "Ensemble":
            return 0.5 * sum(self.fitted[target][m].predict(X) for m in ENSEMBLE)
        return self.fitted[target][name].predict(X)

    def targets(self):
        return list(self.fitted)

    def save(self, path):
        joblib.dump(self.fitted, path)

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))


def cross_validate(train: pd.DataFrame, targets, size: int, folds: int = 5, seed: int = 42):
    """Development-only K-fold CV. Returns fold scores, OOF predictions and fold indices."""
    kf = KFold(folds, shuffle=True, random_state=seed)
    split = list(kf.split(train))
    rows, oof_rows = [], []
    for target in targets:
        X, y = train[FEATURES], train[target].to_numpy()
        oof = {}
        for name, model in model_zoo(size).items():
            pred = np.full(len(y), np.nan)
            for fold, (a, b) in enumerate(split):
                pred[b] = _fit(model, X.iloc[a], y[a]).predict(X.iloc[b])
                rows.append(dict(target=target, model=name, fold=fold, **metrics(y[b], pred[b])))
            oof[name] = pred
        oof["Ensemble"] = 0.5 * (oof["CatBoost"] + oof["XGBoost"])
        for fold, (_, b) in enumerate(split):
            rows.append(dict(target=target, model="Ensemble", fold=fold,
                             **metrics(y[b], oof["Ensemble"][b])))
        for name, pred in oof.items():
            oof_rows.append(pd.DataFrame(dict(setting_id=train.setting_id, target=target,
                                              model=name, observed=y, predicted=pred)))
    folds_df = pd.DataFrame(rows)
    summary = (folds_df.groupby(["target", "model"])
               .agg(R2_mean=("R2", "mean"), R2_SD=("R2", "std"), RMSE_mean=("RMSE", "mean"),
                    RMSE_SD=("RMSE", "std"), MAE_mean=("MAE", "mean")).reset_index())
    return folds_df, pd.concat(oof_rows, ignore_index=True), summary


def fit_all(train: pd.DataFrame, targets, size: int, names=None) -> Surrogate:
    zoo = model_zoo(size)
    names = names or list(zoo)
    fitted = {t: {n: _fit(zoo[n], train[FEATURES], train[t].to_numpy()) for n in names}
              for t in targets}
    return Surrogate(fitted)


def paired_bootstrap(delta, B=4000, seed=42):
    delta = np.asarray(delta, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (B, len(delta)))
    return np.percentile(delta[idx].mean(axis=1), [2.5, 97.5])


def bootstrap_ci(values, stat=np.mean, B=2000, seed=42):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return np.percentile([stat(values[rng.integers(0, len(values), len(values))])
                          for _ in range(B)], [2.5, 97.5])

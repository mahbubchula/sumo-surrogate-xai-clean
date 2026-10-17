"""Prespecified adaptation of the 6x6 surrogate to an 8x8 grid (no test-based selection).

Methods (all reported):
* Frozen_source       - the 6x6 ensemble unchanged;
* Affine_adaptation   - linear recalibration of the source prediction on target data;
* Residual_adaptation - CatBoost learns target-minus-source residuals;
* Target_only         - CatBoost trained on target data alone.
Intervals: split conformal (absolute residuals) calibrated on the target calibration set.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ALPHA, FEATURES
from .surrogates import metrics, paired_bootstrap
from .validation import wilson

METHODS = ["Frozen_source", "Affine_adaptation", "Residual_adaptation", "Target_only"]
BUDGETS = [16, 32, 64, 128]


def _cat(size):
    return CatBoostRegressor(iterations=size, depth=4, learning_rate=.05, random_seed=1901,
                             thread_count=1, verbose=False, allow_writing_files=False)


def _fit(name, X, y, p, size):
    if name == "Frozen_source":
        return None
    if name == "Affine_adaptation":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(p.reshape(-1, 1), y)
    return _cat(size).fit(X, y - p if name == "Residual_adaptation" else y)


def _predict(name, model, X, p):
    if name == "Frozen_source":
        return p
    if name == "Affine_adaptation":
        return model.predict(p.reshape(-1, 1))
    v = model.predict(X)
    return v + p if name == "Residual_adaptation" else v


def adapt(source, train, calib, test, targets, size=400):
    rows, learn, ints, paired, preds = [], [], [], [], []
    for t in targets:
        X = [d[FEATURES] for d in (train, calib, test)]
        y = [d[t].to_numpy() for d in (train, calib, test)]
        p = [source.predict(x, t) for x in X]
        for budget in [b for b in BUDGETS if b <= len(train)]:
            for m in METHODS:
                oof = np.empty(budget)
                for a, b in KFold(4, shuffle=True, random_state=1931).split(X[0].iloc[:budget]):
                    mod = _fit(m, X[0].iloc[a], y[0][a], p[0][a], size)
                    oof[b] = _predict(m, mod, X[0].iloc[b], p[0][b])
                learn.append(dict(target=t, method=m, n_adaptation=budget, **metrics(y[0][:budget], oof)))
        for m in METHODS:
            mod = _fit(m, X[0], y[0], p[0], size)
            pc, pt = _predict(m, mod, X[1], p[1]), _predict(m, mod, X[2], p[2])
            rows.append(dict(target=t, method=m, **metrics(y[2], pt)))
            s = np.sort(np.abs(y[1] - pc))
            rank = math.ceil((len(s) + 1) * (1 - ALPHA))
            q = float(s[rank - 1]) if rank <= len(s) else float("inf")
            cov = (y[2] >= pt - q) & (y[2] <= pt + q)
            ci = wilson(int(cov.sum()), len(cov))
            ints.append(dict(target=t, method=m, coverage=cov.mean(), CI_low=ci[0], CI_high=ci[1],
                             width=2 * q, n_cal=len(s), n_test=len(cov)))
            d = np.abs(y[2] - pt) - np.abs(y[2] - p[2])
            lo, hi = paired_bootstrap(d)
            paired.append(dict(target=t, method=m, MAE_difference_vs_frozen=d.mean(), CI_low=lo, CI_high=hi))
            preds.append(pd.DataFrame(dict(setting_id=test.setting_id, target=t, method=m,
                                           observed=y[2], predicted=pt)))
    return dict(test_metrics=pd.DataFrame(rows), learning=pd.DataFrame(learn),
                intervals=pd.DataFrame(ints), paired=pd.DataFrame(paired),
                predictions=pd.concat(preds, ignore_index=True))

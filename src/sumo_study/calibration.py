"""Surrogate-assisted inverse calibration against EXCLUDED synthetic targets.

Each target is a new parameter vector (never used for training/calibration/testing), run under
several seeds; its seed-mean outputs are the "observations". Two objectives are compared:

* ``aggregate`` - mean time loss and VKT only (two-output objective);
* ``extended``  - adds completed trips and the hourly arrival profile.

Calibrated parameters are re-simulated with FRESH seeds and compared on outputs that were and
were not part of the objective. Equifinality is examined with a dense surrogate scan.
"""
from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from .config import BOUNDS, FEATURES, HIGH, HOURS, LOW
from .design import random_design
from .surrogates import Surrogate

AGG = ["mean_time_loss_sec", "total_vkt_km"]  # two-output aggregate objective
OBJECTIVES = {
    "aggregate": AGG,
    "extended": AGG + ["total_completed_trips"] + [f"arrivals_hour_{h}" for h in HOURS],
}
TOLERANCE_RMS = 0.05  # near-optimal: root-mean-square relative mismatch <= 5 %
VALIDATION_METRICS = ["mean_time_loss_sec", "total_vkt_km", "general_time_loss_sec", "total_completed_trips",
                      "p90_time_loss_sec", "type_default_completed", "unfinished_trips",
                      "mean_waiting_sec"] + [f"arrivals_hour_{h}" for h in HOURS]


def _loss(pred: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Root-mean-square relative mismatch across outputs (columns)."""
    rel = (pred - obs) / np.maximum(np.abs(obs), 1e-6)
    return np.sqrt(np.mean(rel ** 2, axis=-1))


def predict_matrix(s: Surrogate, X: pd.DataFrame, outputs) -> np.ndarray:
    return np.column_stack([s.predict(X, t) for t in outputs])


def calibrate(s: Surrogate, obs: dict, objective: str, trials: int, seed: int):
    outputs = OBJECTIVES[objective]
    target = np.array([obs[t] for t in outputs])
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def f(trial):
        x = pd.DataFrame([{k: trial.suggest_float(k, *BOUNDS[k]) for k in FEATURES}])
        return float(_loss(predict_matrix(s, x, outputs), target)[0])

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(f, n_trials=trials, show_progress_bar=False)
    return study.best_params, study.best_value, study.trials_dataframe()


def equifinality_scan(s: Surrogate, obs: dict, objective: str, n=100_000, seed=0):
    """Dense uniform scan: which parameter regions reproduce the targets equally well?"""
    outputs = OBJECTIVES[objective]
    target = np.array([obs[t] for t in outputs])
    X = random_design(n, seed)
    loss = _loss(predict_matrix(s, X, outputs), target)
    near = X[loss <= TOLERANCE_RMS]
    u = (near.to_numpy() - LOW) / (HIGH - LOW)
    return dict(n_scanned=n, n_near_optimal=len(near), fraction_near_optimal=len(near) / n,
                min_loss=float(loss.min())), pd.DataFrame(
        dict(parameter=FEATURES,
             normalized_range=(u.max(0) - u.min(0)) if len(near) else np.nan,
             normalized_IQR=(np.quantile(u, .75, 0) - np.quantile(u, .25, 0)) if len(near) else np.nan))

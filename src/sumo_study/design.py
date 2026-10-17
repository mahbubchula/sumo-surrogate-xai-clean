"""Experimental designs, job construction and partition-separation checks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import FEATURES, HIGH, LOW


def sobol_design(n: int, seed: int) -> pd.DataFrame:
    """Scrambled Sobol design (n must be a power of two to keep balance properties)."""
    if n & (n - 1):
        raise ValueError("Sobol sample count must be a power of two")
    u = qmc.Sobol(d=len(FEATURES), scramble=True, seed=seed).random_base2(int(math.log2(n)))
    return pd.DataFrame(qmc.scale(u, LOW, HIGH), columns=FEATURES)


def random_design(n: int, seed: int) -> pd.DataFrame:
    """Independent uniform design over the same physical domain (validation partitions)."""
    u = np.random.default_rng(seed).random((n, len(FEATURES)))
    return pd.DataFrame(qmc.scale(u, LOW, HIGH), columns=FEATURES)


def params(row) -> dict:
    return {k: float(row[k]) for k in FEATURES}


def jobs(frame: pd.DataFrame, seed_base: int, tag: str, reps: int = 1, **meta) -> list[dict]:
    """One job per (setting, replicate). Seeds are ``seed_base + i * reps + r`` (disjoint)."""
    out = []
    for i, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        for r in range(reps):
            out.append(dict(params=params(row), seed=seed_base + i * reps + r,
                            meta=dict(setting_id=f"{tag}_{i:04d}", replicate=r, split=tag, **meta)))
    return out


def assert_disjoint(named: dict[str, pd.DataFrame]):
    """No parameter vector may appear in two partitions."""
    sets = {k: set(map(tuple, np.round(v[FEATURES].to_numpy(), 10))) for k, v in named.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                raise AssertionError(f"{len(shared)} settings shared by '{a}' and '{b}'")


def min_distance(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Normalised nearest-neighbour distance from each row of ``a`` to ``b`` (design coverage)."""
    ua = (a[FEATURES].to_numpy() - LOW) / (HIGH - LOW)
    ub = (b[FEATURES].to_numpy() - LOW) / (HIGH - LOW)
    d = np.sqrt(((ua[:, None, :] - ub[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)

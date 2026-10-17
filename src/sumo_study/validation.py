"""Independent testing, simulation-noise analysis and predictive-interval validation.

Estimands are kept explicit:
* ``single_run``  - the outcome of ONE future SUMO run at a new setting (primary);
* ``new_runs``    - every independent run at the test settings (same estimand, more draws);
* ``seed_mean``   - the mean response over seeds (a different, smoother target).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import ALPHA, FEATURES
from .surrogates import MODEL_ORDER, Surrogate, metrics, paired_bootstrap


def wilson(k, n, z=1.959963984540054):
    if not n:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(mid - half), float(mid + half)


def cluster_bootstrap_mean(values: pd.Series, groups: pd.Series, B=4000, seed=42):
    """CI for a mean when observations are clustered (runs within settings)."""
    df = pd.DataFrame(dict(v=values.to_numpy(float), g=groups.to_numpy()))
    agg = df.groupby("g").v.agg(["sum", "count"])
    s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (B, len(s)))
    return np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])


# ----------------------------------------------------------------------------------------
# Simulation noise
# ----------------------------------------------------------------------------------------
def noise_components(runs: pd.DataFrame, targets, label: str) -> pd.DataFrame:
    """Within-setting (seed) vs between-setting variance and the implied accuracy ceiling."""
    rows = []
    for t in targets:
        g = runs.groupby("setting_id")[t]
        k = g.size().mean()
        within = float(g.var(ddof=1).mean())
        between_obs = float(g.mean().var(ddof=1))
        between = max(0.0, between_obs - within / k)
        total = within + between
        rows.append(dict(source=label, target=t, settings=int(g.ngroups), seeds_per_setting=k,
                         within_SD=np.sqrt(within), between_SD=np.sqrt(between),
                         noise_share=within / total if total else np.nan,
                         R2_ceiling_single_run=1 - within / total if total else np.nan,
                         RMSE_floor_single_run=np.sqrt(within),
                         ICC=between / total if total else np.nan))
    return pd.DataFrame(rows)


def replication_summary(runs: pd.DataFrame, surrogate: Surrogate, targets) -> pd.DataFrame:
    rows = []
    extra = ["total_completed_trips", "unfinished_trips", "stat_teleports_total"]
    for sid, g in runs.groupby("setting_id"):
        for t in list(targets) + extra:
            v = g[t].to_numpy(float)
            row = dict(setting_id=sid, metric=t, n=len(v), mean=v.mean(), SD=v.std(ddof=1),
                       CV_pct=100 * v.std(ddof=1) / abs(v.mean()) if v.mean() else np.nan,
                       min=v.min(), max=v.max())
            if t in targets:
                row["surrogate_prediction"] = float(surrogate.predict(g.iloc[[0]][FEATURES], t)[0])
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Frozen-model independent test
# ----------------------------------------------------------------------------------------
def independent_test(surrogate: Surrogate, test_runs: pd.DataFrame, targets):
    """Score frozen models on untouched test settings, for both estimands."""
    single = test_runs[test_runs.replicate == 0].sort_values("setting_id").reset_index(drop=True)
    means = (test_runs.groupby("setting_id")[list(targets)].mean()
             .reindex(single.setting_id).reset_index())
    rows, preds, paired = [], [], []
    for t in targets:
        p_all = {m: surrogate.predict(single[FEATURES], t, m) for m in MODEL_ORDER}
        y1, ym = single[t].to_numpy(), means[t].to_numpy()
        for m, p in p_all.items():
            rows.append(dict(target=t, model=m, estimand="single_run", n=len(y1), **metrics(y1, p)))
            rows.append(dict(target=t, model=m, estimand="seed_mean", n=len(ym), **metrics(ym, p)))
            preds.append(pd.DataFrame(dict(setting_id=single.setting_id, target=t, model=m,
                                           observed=y1, observed_seed_mean=ym, predicted=p)))
            for ref in ["Ridge", "Ensemble"]:
                if ref == m:
                    continue
                delta = np.abs(y1 - p) - np.abs(y1 - p_all[ref])
                lo, hi = paired_bootstrap(delta)
                paired.append(dict(target=t, model=m, reference=ref,
                                   MAE_difference=float(delta.mean()), CI_low=lo, CI_high=hi))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(paired)


# ----------------------------------------------------------------------------------------
# Predictive intervals: fit on train, calibrate on calibration, evaluate on test only
# ----------------------------------------------------------------------------------------
def _conformal_quantile(scores, alpha=ALPHA):
    scores = np.sort(np.asarray(scores, float))
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return (float(scores[rank - 1]) if rank <= len(scores) else float("inf")), rank


class IntervalMethods:
    """Four interval procedures for one target, all fitted without calibration/test data.

    CQR       - LightGBM quantile regression, conformalised (Romano et al., 2019).
    SplitCP   - symmetric split conformal on the ensemble's absolute residuals.
    GP        - native Gaussian-process 90 % interval (model-based, not conformal).
    GP-CP     - split conformal with GP-normalised residuals (locally adaptive).
    """

    names = ["CQR", "SplitCP", "GP", "GP-CP"]

    def __init__(self, train, target, surrogate: Surrogate, size=500, alpha=ALPHA):
        self.t, self.s, self.alpha = target, surrogate, alpha
        X, y = train[FEATURES], train[target]
        common = dict(n_estimators=size, learning_rate=.04, num_leaves=12, min_child_samples=5,
                      verbosity=-1, random_state=42, n_jobs=1, objective="quantile")
        self.q_lo = LGBMRegressor(alpha=alpha / 2, **common).fit(X, y)
        self.q_hi = LGBMRegressor(alpha=1 - alpha / 2, **common).fit(X, y)
        self.z = 1.6448536269514722  # two-sided 90 %
        self.qhat = {}

    def _gp(self, X):
        pipe = self.s.fitted[self.t]["GaussianProcess"]
        mu, sd = pipe[-1].predict(pipe[:-1].transform(X[FEATURES]), return_std=True)
        return mu, np.maximum(sd, 1e-9)

    def _raw(self, X, name):
        if name == "CQR":
            a, b = self.q_lo.predict(X[FEATURES]), self.q_hi.predict(X[FEATURES])
            return np.minimum(a, b), np.maximum(a, b), None
        if name == "SplitCP":
            m = self.s.predict(X[FEATURES], self.t)
            return m, m, None
        mu, sd = self._gp(X)
        if name == "GP":
            return mu - self.z * sd, mu + self.z * sd, None
        return mu, mu, sd

    def calibrate(self, cal):
        y = cal[self.t].to_numpy()
        self.scores = {}
        for name in self.names:
            lo, hi, sd = self._raw(cal, name)
            if name == "GP":
                self.qhat[name], rank = 0.0, None
                self.scores[name] = np.maximum(lo - y, y - hi)
                continue
            s = np.maximum(lo - y, y - hi) if name == "CQR" else np.abs(y - lo)
            if sd is not None:
                s = s / sd
            self.scores[name] = s
            q, rank = _conformal_quantile(s, self.alpha)
            self.qhat[name] = max(0.0, q) if name == "CQR" else q
        return self

    def interval(self, X, name):
        lo, hi, sd = self._raw(X, name)
        q = self.qhat[name]
        if sd is not None:
            return lo - q * sd, hi + q * sd
        return lo - q, hi + q


def interval_score(y, lo, hi, alpha=ALPHA):
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def evaluate_intervals(methods: dict, train, test_runs, targets, alpha=ALPHA):
    """Coverage, width and interval score per method, target and estimand."""
    rows, preds, strata = [], [], []
    single = test_runs[test_runs.replicate == 0].sort_values("setting_id").reset_index(drop=True)
    means = test_runs.groupby("setting_id")[list(targets)].mean().reindex(single.setting_id)
    for t in targets:
        cuts = np.quantile(train[t], [1 / 3, 2 / 3])
        for name in IntervalMethods.names:
            im = methods[t]
            lo, hi = im.interval(single, name)
            bounds = pd.DataFrame(dict(setting_id=single.setting_id, lower=lo, upper=hi))
            # (1) single future run (primary) and (3) seed mean
            for est, y in [("single_run", single[t].to_numpy()),
                           ("seed_mean", means[t].to_numpy())]:
                cov = (y >= lo) & (y <= hi)
                ci = wilson(int(cov.sum()), len(cov))
                rows.append(dict(target=t, method=name, estimand=est, n=len(y),
                                 coverage=cov.mean(), CI_low=ci[0], CI_high=ci[1],
                                 mean_width=float(np.mean(hi - lo)),
                                 median_width=float(np.median(hi - lo)),
                                 interval_score=float(interval_score(y, lo, hi, alpha).mean()),
                                 qhat=im.qhat[name]))
            # (2) all independent runs at the test settings, clustered by setting
            allr = test_runs[["setting_id", t]].merge(bounds, on="setting_id")
            cov = (allr[t] >= allr.lower) & (allr[t] <= allr.upper)
            ci = cluster_bootstrap_mean(cov.astype(float), allr.setting_id)
            rows.append(dict(target=t, method=name, estimand="new_runs", n=len(allr),
                             coverage=cov.mean(), CI_low=ci[0], CI_high=ci[1],
                             mean_width=float(np.mean(hi - lo)), median_width=float(np.median(hi - lo)),
                             interval_score=float(interval_score(allr[t].to_numpy(), allr.lower.to_numpy(),
                                                                 allr.upper.to_numpy(), alpha).mean()),
                             qhat=im.qhat[name]))
            y = single[t].to_numpy()
            df = bounds.assign(target=t, method=name, observed=y, covered=(y >= lo) & (y <= hi),
                               width=hi - lo,
                               stratum=pd.cut(y, [-np.inf, *cuts, np.inf], labels=["low", "medium", "high"]))
            preds.append(df)
            for s, g in df.groupby("stratum", observed=True):
                ci = wilson(int(g.covered.sum()), len(g))
                strata.append(dict(target=t, method=name, stratum=str(s), n=len(g),
                                   coverage=g.covered.mean(), CI_low=ci[0], CI_high=ci[1],
                                   mean_width=g.width.mean()))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), pd.DataFrame(strata)

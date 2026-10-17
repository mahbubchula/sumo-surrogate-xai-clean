"""Global sensitivity: direct-SUMO Sobol indices, surrogate Sobol with refit stability,
ensemble SHAP, ALE response curves with refit bands, and descriptive DML projections.

Sobol estimators (radial A/B/AB_i design):
    V    = Var[f(A), f(B)]
    S1_i = mean(f(B) * (f(AB_i) - f(A))) / V                  (Saltelli et al., 2010)
    ST_i = mean((f(A) - f(AB_i))^2) / (2V)                    (Jansen, 1999)
For SUMO, every evaluation carries independent seed noise with variance s2. Noise inflates V
and the Jansen numerator, but not the S1 numerator. The noise-corrected versions are
    S1_i^c = num_S1 / (V - s2),   ST_i^c = (num_ST/2 - s2) / (V - s2),
with s2 estimated from independent replicate runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, qmc
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from .config import FEATURES, HIGH, LOW
from .surrogates import ENSEMBLE, Surrogate

D = len(FEATURES)


def radial_matrices(power: int, seed: int):
    base = qmc.Sobol(d=2 * D, scramble=True, seed=seed).random_base2(power)
    A = LOW + base[:, :D] * (HIGH - LOW)
    B = LOW + base[:, D:] * (HIGH - LOW)
    AB = []
    for i in range(D):
        m = A.copy()
        m[:, i] = B[:, i]
        AB.append(m)
    return A, B, AB


def stack(A, B, AB) -> pd.DataFrame:
    """Rows: block 'A', 'B', then 'AB_<i>', each with the base-row index j."""
    frames = [pd.DataFrame(A, columns=FEATURES).assign(block="A"),
              pd.DataFrame(B, columns=FEATURES).assign(block="B")]
    frames += [pd.DataFrame(m, columns=FEATURES).assign(block=f"AB_{i}") for i, m in enumerate(AB)]
    out = pd.concat(frames, ignore_index=True)
    out["j"] = np.tile(np.arange(len(A)), D + 2)
    return out


def sobol_indices(fA, fB, fAB, noise_var=0.0, B=1000, seed=0):
    """Point estimates and row-bootstrap CIs; fAB has shape (D, N)."""
    n = len(fA)

    def est(ix):
        a, b, ab = fA[ix], fB[ix], fAB[:, ix]
        V = np.var(np.concatenate([a, b]), ddof=1)
        Vc = max(V - noise_var, 1e-12)
        s1 = np.mean(b * (ab - a), axis=1) / Vc
        st = (0.5 * np.mean((a - ab) ** 2, axis=1) - noise_var) / Vc
        return s1, st, V

    s1, st, V = est(np.arange(n))
    rng = np.random.default_rng(seed)
    boot = [est(rng.integers(0, n, n)) for _ in range(B)]
    s1b = np.array([b[0] for b in boot])
    stb = np.array([b[1] for b in boot])
    return pd.DataFrame(dict(parameter=FEATURES, S1=s1, S1_low=np.percentile(s1b, 2.5, 0),
                             S1_high=np.percentile(s1b, 97.5, 0), ST=st,
                             ST_low=np.percentile(stb, 2.5, 0), ST_high=np.percentile(stb, 97.5, 0),
                             total_variance=V, noise_variance=noise_var))


def evaluate_stacked(stacked: pd.DataFrame, values: np.ndarray):
    n = int(stacked.j.max()) + 1
    v = values.reshape(D + 2, n)
    return v[0], v[1], v[2:]


def surrogate_sobol(predict, power: int, seed: int, B=300):
    A, Bm, AB = radial_matrices(power, seed)
    st = stack(A, Bm, AB)
    fA, fB, fAB = evaluate_stacked(st, predict(st[FEATURES]))
    return sobol_indices(fA, fB, fAB, 0.0, B=B, seed=seed)


def refit_ensemble(surrogate: Surrogate, target, X, y):
    models = [clone(surrogate.fitted[target][m]).fit(X, y) for m in ENSEMBLE]
    return lambda Z: 0.5 * sum(m.predict(pd.DataFrame(Z, columns=FEATURES)) for m in models)


def rank_stability(indices: pd.DataFrame, reference: pd.DataFrame, col="ST"):
    rows = []
    ref = reference.set_index("parameter")[col].reindex(FEATURES)
    for b, g in indices.groupby("replicate"):
        v = g.set_index("parameter")[col].reindex(FEATURES)
        top2 = set(v.nlargest(2).index)
        rows.append(dict(replicate=b, kendall_tau=kendalltau(ref, v).statistic,
                         top2_same=top2 == set(ref.nlargest(2).index), top1=v.idxmax()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
def ale_curve(predict, X: pd.DataFrame, feature: str, bins: int = 10):
    """First-order ALE with quantile bins, mid-bin accumulation and count-weighted centring."""
    edges = np.unique(np.quantile(X[feature], np.linspace(0, 1, bins + 1)))
    local, counts = [], []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        m = (X[feature] >= edges[i]) & ((X[feature] <= edges[i + 1]) if last else (X[feature] < edges[i + 1]))
        sub = X.loc[m]
        counts.append(len(sub))
        if not len(sub):
            local.append(0.0)
            continue
        a, b = sub.copy(), sub.copy()
        a[feature], b[feature] = edges[i], edges[i + 1]
        local.append(float(np.mean(predict(b) - predict(a))))
    acc = np.concatenate([[0.0], np.cumsum(local)])
    mid = 0.5 * (acc[:-1] + acc[1:])
    centre = np.average(mid, weights=np.maximum(counts, 1e-12))
    return edges, acc - centre, np.array(counts)


def ale_with_refits(surrogate, target, train, n_boot, seed=2026, bins=10):
    X = train[FEATURES]
    rows = []
    preds = [(-1, lambda Z: surrogate.predict(Z, target))]
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        ix = rng.integers(0, len(X), len(X))
        preds.append((b, refit_ensemble(surrogate, target, X.iloc[ix], train[target].iloc[ix])))
    for b, f in preds:
        for k in FEATURES:
            edges, curve, counts = ale_curve(f, X, k, bins)
            rows.extend(dict(target=target, replicate=b, parameter=k, x=float(edges[i]),
                             ALE=float(curve[i]), bin_count=int(counts[i - 1]) if i else int(counts[0]))
                        for i in range(len(edges)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
def ensemble_shap(surrogate: Surrogate, target, X: pd.DataFrame, background_n=100):
    """Interventional TreeSHAP for both ensemble members on a shared background, averaged."""
    import shap
    bg = X.sample(n=min(background_n, len(X)), random_state=2026)
    exps = [shap.TreeExplainer(surrogate.fitted[target][m], data=bg,
                               feature_perturbation="interventional")(X) for m in ENSEMBLE]
    cat = exps[ENSEMBLE.index("CatBoost")]
    model = surrogate.fitted[target]["CatBoost"]
    resid = model.predict(X) - (np.asarray(cat.base_values) + np.asarray(cat.values).sum(1))
    corrected = False
    if not np.allclose(resid, 0, rtol=1e-4, atol=1e-3):
        scale, bias = model.get_scale_and_bias()
        if not (np.isclose(scale, 1) and np.allclose(resid, bias, rtol=1e-4, atol=1e-3)):
            raise RuntimeError("Unexpected CatBoost SHAP residual")
        cat.base_values = np.asarray(cat.base_values) + bias
        corrected = True
    values = 0.5 * sum(np.asarray(e.values) for e in exps)
    base = 0.5 * sum(np.asarray(e.base_values) for e in exps)
    err = np.abs(base + values.sum(1) - surrogate.predict(X, target))
    if err.max() > 1e-3 * max(1.0, np.abs(base).mean()):
        raise RuntimeError(f"Ensemble SHAP additivity failed (max error {err.max():.2e})")
    return values, dict(background_rows=len(bg), catboost_bias_corrected=corrected,
                        max_additivity_error=float(err.max()))


def dml_projection(train, target, size=250, folds=5):
    """Cross-fitted partially linear projection per input (descriptive, not causal)."""
    from catboost import CatBoostRegressor
    x = (train[FEATURES] - train[FEATURES].mean()) / train[FEATURES].std(ddof=1)
    y = train[target].to_numpy()
    kf = KFold(folds, shuffle=True, random_state=42)
    rows = []
    for k in FEATURES:
        w, t = x.drop(columns=k), x[k].to_numpy()
        m = CatBoostRegressor(iterations=size, depth=4, verbose=False, random_seed=42,
                              thread_count=1, allow_writing_files=False)
        ry = y - cross_val_predict(m, w, y, cv=kf)
        rt = t - cross_val_predict(Ridge(alpha=1), w, t, cv=kf)
        beta = float(rt @ ry / (rt @ rt))
        psi = (ry - beta * rt) * rt
        se = float(np.sqrt(np.mean(psi ** 2) / (len(y) * np.mean(rt ** 2) ** 2)))
        sd = float(train[k].std(ddof=1))
        rows.append(dict(target=target, parameter=k, estimate_per_SD=beta, SE=se,
                         CI_low=beta - 1.96 * se, CI_high=beta + 1.96 * se, input_SD=sd))
    return pd.DataFrame(rows)

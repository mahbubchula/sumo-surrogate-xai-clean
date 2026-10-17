"""Surrogate-proposed interventions (recourse) validated in SUMO with matched seeds.

Three arms per scenario share the same SUMO seeds:
* ``baseline``  - the congested baseline setting;
* ``recourse``  - minimum-change lever setting predicted to cut delay >= 15 % while keeping
  predicted completed trips >= 95 % of baseline;
* ``random``    - a random lever change of the SAME normalised distance (control: does the
  surrogate's direction matter, or would any change of that size do?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMPLETION_GUARD, DELAY_TARGET, FEATURES, HIGH, LEVERS, LOW, TARGETS
from .design import params
from .surrogates import Surrogate

IDX = [FEATURES.index(c) for c in LEVERS]
SPAN = HIGH[IDX] - LOW[IDX]


def search(surrogate: Surrogate, baseline: dict, budget: int, seed: int):
    x0 = np.array([baseline[k] for k in FEATURES])
    u0 = (x0[IDX] - LOW[IDX]) / SPAN
    rng = np.random.default_rng(seed)
    u = np.vstack([u0, rng.random((budget // 2, len(IDX))),
                   np.clip(u0 + rng.normal(0, .15, (budget - budget // 2, len(IDX))), 0, 1)])
    xs = np.tile(x0, (len(u), 1))
    xs[:, IDX] = LOW[IDX] + u * SPAN
    X = pd.DataFrame(xs, columns=FEATURES)
    delay = surrogate.predict(X, TARGETS[0])
    trips = surrogate.predict(X, "total_completed_trips")
    cost = np.sqrt(((u - u0) ** 2).sum(1))
    ok = (delay <= (1 - DELAY_TARGET) * delay[0]) & (trips >= COMPLETION_GUARD * trips[0])
    i = int(np.where(ok)[0][np.argmin(cost[ok])]) if ok.any() else int(np.argmin(delay))
    rec = params(pd.Series(xs[i], index=FEATURES))
    # Random control: same distance, random direction, projected into the box.
    rand = None
    for _ in range(1000):
        d = rng.normal(size=len(IDX))
        d /= np.linalg.norm(d)
        uc = u0 + cost[i] * d
        if np.all((uc >= 0) & (uc <= 1)):
            rand = x0.copy()
            rand[IDX] = LOW[IDX] + uc * SPAN
            break
    if rand is None:  # fall back to clipping (recorded)
        rand = x0.copy()
        rand[IDX] = LOW[IDX] + np.clip(u0 + cost[i] * d, 0, 1) * SPAN
    rand = params(pd.Series(rand, index=FEATURES))
    pr = surrogate.predict(pd.DataFrame([rand]), TARGETS[0])[0]
    return rec, rand, dict(
        predicted_baseline=float(delay[0]), predicted_recourse=float(delay[i]),
        predicted_random=float(pr), predicted_reduction_pct=100 * (1 - delay[i] / delay[0]),
        predicted_random_reduction_pct=100 * (1 - pr / delay[0]),
        predicted_trip_ratio=float(trips[i] / trips[0]), normalized_distance=float(cost[i]),
        surrogate_feasible=bool(ok[i]), feasible_candidates=int(ok.sum()), candidates=len(u))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Achieved reduction with matched-seed (paired) bootstrap CI for each arm."""
    out = []
    rng = np.random.default_rng(42)
    for sid, g in runs.groupby("setting_id"):
        base = g[g.arm == "baseline"].sort_values("seed")
        for arm in ["recourse", "random"]:
            b = g[g.arm == arm].sort_values("seed")
            assert list(base.seed) == list(b.seed), "seeds must be matched"
            av, bv = base[TARGETS[0]].to_numpy(), b[TARGETS[0]].to_numpy()
            idx = rng.integers(0, len(av), (4000, len(av)))
            boot = 100 * (1 - bv[idx].mean(1) / av[idx].mean(1))
            red = 100 * (1 - bv.mean() / av.mean())
            ratio = b.total_completed_trips.mean() / base.total_completed_trips.mean()
            vkt = b.total_vkt_km.mean() / base.total_vkt_km.mean()
            gen = 100 * (1 - b.general_time_loss_sec.mean() / base.general_time_loss_sec.mean())
            gb = 100 * (1 - b.general_time_loss_sec.to_numpy()[idx].mean(1)
                        / base.general_time_loss_sec.to_numpy()[idx].mean(1))
            out.append(dict(
                setting_id=sid, arm=arm, n_seed_pairs=len(av), baseline_delay=av.mean(),
                arm_delay=bv.mean(), achieved_reduction_pct=red,
                CI_low=np.percentile(boot, 2.5), CI_high=np.percentile(boot, 97.5),
                seeds_improved=int((bv < av).sum()), completion_ratio=ratio, vkt_ratio=vkt,
                general_reduction_pct=gen, general_CI_low=np.percentile(gb, 2.5),
                general_CI_high=np.percentile(gb, 97.5),
                baseline_served=base.served_fraction.mean(), arm_served=b.served_fraction.mean(),
                bus_vehicles_ratio=b.demand_bus_in_window.mean() / max(base.demand_bus_in_window.mean(), 1),
                private_trip_ratio=b.type_default_completed.mean() / base.type_default_completed.mean(),
                baseline_unfinished=base.unfinished_trips.mean(), arm_unfinished=b.unfinished_trips.mean(),
                delay_target_met=bool(red >= 100 * DELAY_TARGET),
                completion_guard_met=bool(ratio >= COMPLETION_GUARD),
                joint_met=bool(red >= 100 * DELAY_TARGET and ratio >= COMPLETION_GUARD)))
    return pd.DataFrame(out)

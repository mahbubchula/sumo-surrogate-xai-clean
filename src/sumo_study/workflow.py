"""Ordered, resumable research workflow. Each stage is one analysis of the article.

Usage (notebook or CLI)::

    study = Study("results", profile="full", workers=8)
    study.run("design"); study.run("models"); ...   # or study.run_all()

Rules enforced in code:
* models are frozen (hashes recorded) before any calibration or test outcome is read;
* calibration data are used only to calibrate intervals, test data only to evaluate;
* any failed simulation stops the analysis (no silent exclusion);
* rerunning a stage marks its downstream stages stale.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from . import calibration as cal
from . import design as dz
from . import policy, sensitivity as sa, validation as val
from .config import (AUX_TARGETS, DESIGN_SEEDS, FEATURES, HIGH, LEVERS, LOW, PHYSICS, PROFILES,
                     RUN_SEED_BASE, TARGETS)
from .io import atomic_json, file_sha256, read_json, save_csv, stable_hash
from .simulator import Simulator
from .surrogates import ENSEMBLE, MODEL_ORDER, Surrogate, cross_validate, fit_all, metrics

STAGES = {  # stage: prerequisites
    "design": [],
    "models": ["design"],
    "independent_test": ["models"],
    "intervals": ["models"],
    "replication": ["models"],
    "recourse": ["models"],
    "inverse_calibration": ["models"],
    "mechanism": ["design"],
    "sobol_sumo": ["replication"],
    "sensitivity": ["models", "sobol_sumo"],
    "learning_curve": ["models", "replication"],
    "teleport": ["design"],
    "routing": ["design"],
    "transfer": ["models", "intervals"],
    "runtime_quality": ["design", "replication", "recourse", "inverse_calibration", "mechanism",
                        "sobol_sumo", "teleport", "transfer"],
    "figures": [],
    "export": [],
}
STAGES["figures"] = [s for s in STAGES if s not in ("figures", "export")]
STAGES["export"] = ["figures"]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "catboost", "xgboost", "lightgbm",
            "SALib", "shap", "optuna", "matplotlib", "joblib"]


def _descendants(stage):
    out = {stage}
    while True:
        new = {s for s, deps in STAGES.items() if any(d in out for d in deps)} - out
        if not new:
            return out - {stage}
        out |= new


class Study:
    def __init__(self, output="results", profile="smoke", workers=4, cache=None):
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {list(PROFILES)}")
        self.profile, self.cfg = profile, dict(PROFILES[profile])
        self.root = Path(output).resolve() / profile
        self.cache = Path(cache).resolve() if cache else Path(output).resolve().parent / "cache"
        for d in ["tables", "figures", "designs", "models", "logs"]:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.sim = Simulator(self.cache, workers=workers)
        self.environment = dict(
            pipeline=__version__, python=sys.version.split()[0], platform=platform.platform(),
            sumo=self.sim.version, simulator_fingerprint=self.sim.fingerprint,
            packages={p: self._v(p) for p in PACKAGES}, profile=profile, config=self.cfg,
            physics=PHYSICS, design_seeds=DESIGN_SEEDS, run_seed_bases=RUN_SEED_BASE)
        man = self.root / "manifest.json"
        if man.exists():
            old = read_json(man)
            if old.get("simulator_fingerprint") != self.sim.fingerprint or old.get("config") != self.cfg:
                raise RuntimeError("Simulator or experiment configuration changed. Use a new "
                                   "output folder; existing results are protected.")
        atomic_json(man, self.environment)
        self.status_file = self.root / "stage_status.json"
        self.status = read_json(self.status_file) if self.status_file.exists() else {}
        self._surrogate = None

    @staticmethod
    def _v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return "not installed"

    # ---------------- helpers ----------------
    def t(self, name):
        return self.root / "tables" / f"{name}.csv"

    def save(self, df, name):
        save_csv(df, self.t(name))
        return df

    def load(self, name):
        return pd.read_csv(self.t(name))

    def json(self, name, value):
        atomic_json(self.root / "tables" / f"{name}.json", value)

    def simulate(self, jobs, name, sim=None):
        return (sim or self.sim).run(jobs, name, self.root / "tables")

    @property
    def surrogate(self) -> Surrogate:
        if self._surrogate is None:
            self._surrogate = Surrogate.load(self.root / "models" / "frozen_surrogates.joblib")
        return self._surrogate

    def data(self, split):
        return self.load(split)

    # ---------------- execution ----------------
    def run(self, stage):
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage}; choose from {list(STAGES)}")
        missing = [d for d in STAGES[stage] if self.status.get(d, {}).get("status") != "complete"]
        if missing:
            raise RuntimeError(f"Complete prerequisite stages first: {', '.join(missing)}")
        for dep in _descendants(stage):
            if self.status.get(dep, {}).get("status") == "complete":
                self.status[dep] = dict(status="stale", reason=f"upstream '{stage}' rerun")
        start = time.time()
        self.status[stage] = dict(status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(self.status_file, self.status)
        print(f"=== {stage} ({self.profile}) ===", flush=True)
        try:
            out = getattr(self, f"stage_{stage}")()
        except Exception as exc:
            self.status[stage] = dict(status="failed", error=repr(exc),
                                      traceback=traceback.format_exc()[-3000:])
            atomic_json(self.status_file, self.status)
            raise
        self.status[stage] = dict(status="complete", elapsed_sec=round(time.time() - start, 1),
                                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        atomic_json(self.status_file, self.status)
        print(f"COMPLETE {stage}: {time.time() - start:.1f} s", flush=True)
        return out

    def run_all(self, resume=True, until=None):
        for s in STAGES:
            if resume and self.status.get(s, {}).get("status") == "complete":
                print("skip (complete):", s)
            else:
                self.run(s)
            if s == until:
                break

    # =====================================================================================
    # 1. Designs and simulations (development / calibration / test partitions)
    # =====================================================================================
    def stage_design(self):
        c = self.cfg
        designs = dict(train=dz.sobol_design(c["n_train"], DESIGN_SEEDS["train"]),
                       calibration=dz.random_design(c["n_cal"], DESIGN_SEEDS["calibration"]),
                       test=dz.random_design(c["n_test"], DESIGN_SEEDS["test"]))
        dz.assert_disjoint(designs)
        reps = dict(train=1, calibration=1, test=c["test_seeds"])
        out = {}
        for name, x in designs.items():
            save_csv(x, self.root / "designs" / f"{name}.csv")
            out[name] = self.simulate(dz.jobs(x, RUN_SEED_BASE[name], name, reps[name]), name)
        seeds = pd.concat([o.seed for o in out.values()])
        assert seeds.is_unique, "SUMO seeds must be unique across partitions"
        nn = pd.DataFrame(dict(split=["calibration"] * len(designs["calibration"]) + ["test"] * len(designs["test"]),
                               nn_distance_to_train=np.concatenate([
                                   dz.min_distance(designs["calibration"], designs["train"]),
                                   dz.min_distance(designs["test"], designs["train"])])))
        self.save(nn, "design_nearest_neighbour_distance")
        return {k: len(v) for k, v in out.items()}

    # =====================================================================================
    # 2. Development CV and frozen surrogates (only development data are read)
    # =====================================================================================
    def stage_models(self):
        train = self.data("train")
        folds, oof, summary = cross_validate(train, TARGETS, self.cfg["model_size"])
        self.save(folds, "dev_cv_folds")
        self.save(oof, "dev_cv_oof_predictions")
        self.save(summary, "dev_cv_summary")
        s = fit_all(train, TARGETS, self.cfg["model_size"])
        aux = fit_all(train, AUX_TARGETS, self.cfg["model_size"], names=list(ENSEMBLE))
        s.fitted.update(aux.fitted)
        path = self.root / "models" / "frozen_surrogates.joblib"
        s.save(path)
        self._surrogate = None
        atomic_json(self.root / "models" / "freeze_record.json", dict(
            frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"), sha256=file_sha256(path),
            trained_on="train partition only", n_train=len(train),
            primary="equal-weight CatBoost + XGBoost ensemble",
            hyperparameters="fixed a priori; no tuning on calibration or test data",
            targets=TARGETS, auxiliary_targets=AUX_TARGETS))
        return summary

    # =====================================================================================
    # 3. Independent test of frozen surrogates + simulation-noise ceiling
    # =====================================================================================
    def stage_independent_test(self):
        test = self.data("test")
        m, p, paired = val.independent_test(self.surrogate, test, TARGETS)
        self.save(m, "test_metrics")
        self.save(p, "test_predictions")
        self.save(paired, "test_paired_mae_differences")
        if self.cfg["test_seeds"] > 1:
            self.save(val.noise_components(test, TARGETS, "test settings"), "noise_test_settings")
        return m[m.estimand == "single_run"]

    # =====================================================================================
    # 4. Predictive intervals: train -> calibration -> untouched test
    # =====================================================================================
    def stage_intervals(self):
        train, calib, test = self.data("train"), self.data("calibration"), self.data("test")
        methods = {t: val.IntervalMethods(train, t, self.surrogate, self.cfg["model_size"]).calibrate(calib)
                   for t in TARGETS}
        joblib.dump(methods, self.root / "models" / "interval_methods.joblib")
        summary, preds, strata = val.evaluate_intervals(methods, train, test, TARGETS)
        self.save(summary, "interval_coverage")
        self.save(preds, "interval_test_predictions")
        self.save(strata, "interval_coverage_by_stratum")
        scores = pd.concat([pd.DataFrame(dict(target=t, method=n, score=s))
                            for t, im in methods.items() for n, s in im.scores.items()])
        self.save(scores, "interval_calibration_scores")
        return summary

    # =====================================================================================
    # 5. Repeated-seed replication
    # =====================================================================================
    def stage_replication(self):
        train = self.data("train").sort_values(TARGETS[0]).reset_index(drop=True)
        ix = np.linspace(0, len(train) - 1, self.cfg["n_rep"]).round().astype(int)
        x = train.iloc[ix][FEATURES].reset_index(drop=True)
        x["congestion_level"] = pd.qcut(np.arange(len(x)), 3, labels=["low", "medium", "high"]).astype(str)
        save_csv(x, self.root / "designs" / "replication.csv")
        runs = self.simulate(dz.jobs(x[FEATURES], RUN_SEED_BASE["replication"], "rep",
                                     self.cfg["rep_seeds"]), "replication_runs")
        runs = runs.merge(x.assign(setting_id=[f"rep_{i:04d}" for i in range(len(x))])[["setting_id", "congestion_level"]],
                          on="setting_id")
        self.save(runs, "replication_runs")
        self.save(val.replication_summary(runs, self.surrogate, TARGETS), "replication_summary")
        noise = val.noise_components(runs, TARGETS + ["total_completed_trips", "p90_time_loss_sec"],
                                     "replicated development settings")
        return self.save(noise, "noise_replication")

    # =====================================================================================
    # 6. Recourse proposed by the surrogate, validated in SUMO
    # =====================================================================================
    def stage_recourse(self):
        train = self.data("train")
        chosen = train.nlargest(self.cfg["n_recourse"], TARGETS[0]).reset_index(drop=True)
        cands, jobs = [], []
        for i, row in chosen.iterrows():
            base = dz.params(row)
            rec, rnd, info = policy.search(self.surrogate, base, self.cfg["search_budget"], 700 + i)
            sid = f"policy_{i:03d}"
            cands.append(dict(setting_id=sid, source=row.setting_id, **info,
                              **{f"baseline_{k}": base[k] for k in LEVERS},
                              **{f"recourse_{k}": rec[k] for k in LEVERS},
                              **{f"random_{k}": rnd[k] for k in LEVERS}))
            for r in range(self.cfg["policy_seeds"]):
                seed = RUN_SEED_BASE["recourse"] + i * self.cfg["policy_seeds"] + r
                for arm, x in [("baseline", base), ("recourse", rec), ("random", rnd)]:
                    jobs.append(dict(params=x, seed=seed, meta=dict(setting_id=sid, replicate=r, arm=arm)))
        cands = self.save(pd.DataFrame(cands), "recourse_candidates")
        runs = self.simulate(jobs, "recourse_runs")
        summ = policy.summarize(runs).merge(
            cands[["setting_id", "predicted_reduction_pct", "predicted_random_reduction_pct",
                   "normalized_distance", "surrogate_feasible", "predicted_trip_ratio"]], on="setting_id")
        self.save(summ, "recourse_validation")
        agg = (summ.groupby("arm").agg(scenarios=("setting_id", "count"),
                                       mean_achieved=("achieved_reduction_pct", "mean"),
                                       median_achieved=("achieved_reduction_pct", "median"),
                                       delay_target_rate=("delay_target_met", "mean"),
                                       joint_rate=("joint_met", "mean"),
                                       mean_general_reduction=("general_reduction_pct", "mean"),
                                       general_target_rate=("general_reduction_pct", lambda z: (z >= 15).mean()),
                                       mean_completion_ratio=("completion_ratio", "mean")).reset_index())
        rec = summ[summ.arm == "recourse"]
        agg["mean_predicted"] = [rec.predicted_reduction_pct.mean() if a == "recourse"
                                 else rec.predicted_random_reduction_pct.mean() for a in agg.arm]
        return self.save(agg, "recourse_overview")

    # =====================================================================================
    # 7. Inverse calibration on excluded targets with fresh-seed validation
    # =====================================================================================
    def stage_inverse_calibration(self):
        c = self.cfg
        truth = dz.random_design(c["n_targets"], DESIGN_SEEDS["targets"])
        dz.assert_disjoint(dict(targets=truth, train=self.data("train")[FEATURES],
                                calibration=self.data("calibration")[FEATURES],
                                test=self.data("test")[FEATURES]))
        save_csv(truth, self.root / "designs" / "inverse_targets.csv")
        obs_runs = self.simulate(dz.jobs(truth, RUN_SEED_BASE["inverse_target"], "target",
                                         c["target_seeds"]), "inverse_target_runs")
        recs, trials, scans, ranges, jobs = [], [], [], [], []
        for i, (sid, g) in enumerate(obs_runs.groupby("setting_id")):
            obs = g.mean(numeric_only=True).to_dict()
            for o_i, objective in enumerate(cal.OBJECTIVES):
                best, loss, tr = cal.calibrate(self.surrogate, obs, objective, c["calib_trials"], 2026 + i)
                trials.append(tr.assign(setting_id=sid, objective=objective))
                rec = dict(setting_id=sid, objective=objective, loss=loss)
                for k in FEATURES:
                    rec[f"true_{k}"], rec[f"calibrated_{k}"] = float(g.iloc[0][k]), best[k]
                recs.append(rec)
                info, rg = cal.equifinality_scan(self.surrogate, obs, objective,
                                                 n=100_000 if self.profile == "full" else 5000, seed=i)
                scans.append(dict(setting_id=sid, objective=objective, **info))
                ranges.append(rg.assign(setting_id=sid, objective=objective))
                for r in range(c["target_seeds"]):
                    seed = RUN_SEED_BASE["inverse_validation"] + (i * 2 + o_i) * c["target_seeds"] + r
                    jobs.append(dict(params=best, seed=seed,
                                     meta=dict(setting_id=sid, objective=objective, replicate=r)))
        recs = self.save(pd.DataFrame(recs), "inverse_parameters")
        self.save(pd.concat(trials, ignore_index=True), "inverse_trials")
        self.save(pd.DataFrame(scans), "inverse_equifinality")
        self.save(pd.concat(ranges, ignore_index=True), "inverse_equifinality_ranges")
        valr = self.simulate(jobs, "inverse_validation_runs")
        rows = []
        for (sid, obj), g in valr.groupby(["setting_id", "objective"]):
            tgt = obs_runs[obs_runs.setting_id == sid]
            for mtr in cal.VALIDATION_METRICS:
                a, b = float(tgt[mtr].mean()), float(g[mtr].mean())
                sd = float(tgt[mtr].std(ddof=1))
                rows.append(dict(setting_id=sid, objective=objective_label(obj), metric=mtr,
                                 in_objective=mtr in cal.OBJECTIVES[obj], target_mean=a,
                                 calibrated_mean=b, relative_error_pct=100 * abs(b - a) / abs(a) if a else np.nan,
                                 standardized_error=(b - a) / sd if sd else np.nan,
                                 GEH=float(np.sqrt(2 * (b - a) ** 2 / (a + b))) if mtr.startswith("arrivals") and a + b > 0 else np.nan))
        self.save(pd.DataFrame(rows), "inverse_validation")
        rec_rows = []
        for _, r in recs.iterrows():
            for k in FEATURES:
                err = (r[f"calibrated_{k}"] - r[f"true_{k}"]) / (HIGH - LOW)[FEATURES.index(k)]
                rec_rows.append(dict(setting_id=r.setting_id, objective=objective_label(r.objective),
                                     parameter=k, normalized_error=err, abs_error=abs(err)))
        return self.save(pd.DataFrame(rec_rows), "inverse_parameter_recovery")

    # =====================================================================================
    # 8. Controlled headway x departure-variation factorial
    # =====================================================================================
    def stage_mechanism(self):
        c = self.cfg
        L = c["mech_levels"]
        dv = np.linspace(LOW[4], HIGH[4], L)
        bh = np.linspace(LOW[6], HIGH[6], L)
        mid = pd.Series((LOW + HIGH) / 2, index=FEATURES)
        backgrounds = {"midpoint": mid.copy()}
        car = mid.copy()
        car["carRate"], car["carPreference"] = 0.80, 0.85
        backgrounds["car-oriented"] = car
        jobs = []
        for bg_name, bgx in list(backgrounds.items())[:c["mech_backgrounds"]]:
            for i, a in enumerate(dv):
                for j, b in enumerate(bh):
                    x = bgx.copy()
                    x["departureVariation"], x["busHeadway"] = a, b
                    for r in range(c["mech_seeds"]):  # common random numbers across cells
                        jobs.append(dict(params=dz.params(x), seed=RUN_SEED_BASE["mechanism"] + r,
                                         meta=dict(setting_id=f"mech_{bg_name}_{i:02d}_{j:02d}",
                                                   background=bg_name, dv_level=i, headway_level=j,
                                                   replicate=r)))
        runs = self.simulate(jobs, "mechanism_runs")
        outs = ["mean_time_loss_sec", "general_time_loss_sec", "served_fraction", "type_default_delay", "type_bus_delay", "type_random_delay",
                "demand_default_in_window", "demand_bus_in_window", "type_bus_completed",
                "total_completed_trips", "total_vkt_km", "mean_waiting_sec"]
        cell = runs.groupby(["background", "dv_level", "headway_level", "departureVariation",
                             "busHeadway"])[outs].agg(["mean", "std"]).reset_index()
        cell.columns = ["_".join(col).rstrip("_") for col in cell.columns]
        self.save(cell, "mechanism_cells")
        rows = []
        for (bg, i), g in runs.groupby(["background", "dv_level"]):
            lo = g[g.headway_level == 0].sort_values("seed")
            hi = g[g.headway_level == L - 1].sort_values("seed")
            assert list(lo.seed) == list(hi.seed)
            for o in outs:
                d = hi[o].to_numpy(float) - lo[o].to_numpy(float)
                if np.isnan(d).all():
                    continue
                ci = val.cluster_bootstrap_mean(pd.Series(d), pd.Series(np.arange(len(d))))
                rows.append(dict(background=bg, dv_level=i, departureVariation=float(lo.departureVariation.iloc[0]),
                                 outcome=o, contrast=f"headway {bh[-1]:.0f} s minus {bh[0]:.0f} s",
                                 mean_difference=np.nanmean(d), CI_low=ci[0], CI_high=ci[1], n_pairs=len(d)))
        self.save(pd.DataFrame(rows), "mechanism_headway_contrasts")
        # two-way variance decomposition of cell means (descriptive)
        dec = []
        for (bg, g), y in [(bgg, y) for bgg in runs.groupby("background")
                           for y in ["mean_time_loss_sec", "general_time_loss_sec"]]:
            m = g.groupby(["dv_level", "headway_level"])[y].mean().unstack()
            grand = m.values.mean()
            a = m.mean(1) - grand
            b = m.mean(0) - grand
            inter = m.values - grand - a.values[:, None] - b.values[None, :]
            ss = dict(departureVariation=L * (a ** 2).sum(), busHeadway=L * (b ** 2).sum(),
                      interaction=(inter ** 2).sum())
            tot = sum(ss.values())
            dec += [dict(background=bg, outcome=y, component=k, share=v / tot) for k, v in ss.items()]
        return self.save(pd.DataFrame(dec), "mechanism_variance_shares")

    # =====================================================================================
    # 9. Direct-SUMO Sobol indices: simulator ground truth for the surrogate GSA
    # =====================================================================================
    def stage_sobol_sumo(self):
        p = self.cfg["sobol_sim_power"]
        A, B, AB = sa.radial_matrices(p, DESIGN_SEEDS["sobol_sim"])
        st = sa.stack(A, B, AB).reset_index(drop=True)
        jobs = [dict(params=dz.params(r), seed=RUN_SEED_BASE["sobol_sim"] + k,
                     meta=dict(setting_id=f"sob_{k:05d}", block=r.block, j=int(r.j)))
                for k, r in st.iterrows()]
        runs = self.simulate(jobs, "sobol_sumo_runs")
        runs = runs.set_index("setting_id").loc[[f"sob_{k:05d}" for k in range(len(st))]].reset_index()
        noise = self.load("noise_replication").set_index("target")
        rows, sur = [], []
        for t in TARGETS:
            fA, fB, fAB = sa.evaluate_stacked(st, runs[t].to_numpy(float))
            s2 = float(noise.loc[t, "within_SD"] ** 2)
            rows.append(sa.sobol_indices(fA, fB, fAB, 0.0).assign(target=t, estimator="SUMO raw"))
            rows.append(sa.sobol_indices(fA, fB, fAB, s2).assign(target=t, estimator="SUMO noise-corrected"))
            sA, sB, sAB = sa.evaluate_stacked(st, self.surrogate.predict(st[FEATURES], t))
            sur.append(sa.sobol_indices(sA, sB, sAB, 0.0).assign(target=t, estimator="Surrogate, same design"))
        return self.save(pd.concat(rows + sur, ignore_index=True), "sobol_sumo_indices")

    # =====================================================================================
    # 10. Surrogate sensitivity with refit stability, SHAP, ALE, DML
    # =====================================================================================
    def stage_sensitivity(self):
        c = self.cfg
        train = self.data("train")
        X, rng = train[FEATURES], np.random.default_rng(2026)
        idx, refits = [], []
        for t in TARGETS:
            ref = sa.surrogate_sobol(lambda Z: self.surrogate.predict(Z, t), c["saltelli_power"], 2026)
            idx.append(ref.assign(target=t, replicate=-1))
            for b in range(c["n_boot"]):
                ix = rng.integers(0, len(X), len(X))
                f = sa.refit_ensemble(self.surrogate, t, X.iloc[ix], train[t].iloc[ix])
                r = sa.surrogate_sobol(f, min(c["saltelli_power"], 10), 3000 + b, B=50)
                refits.append(r.assign(target=t, replicate=b))
        allidx = pd.concat(idx + refits, ignore_index=True)
        self.save(allidx, "sobol_surrogate_refits")
        stab = []
        for t in TARGETS:
            g = allidx[allidx.target == t]
            rs = sa.rank_stability(g[g.replicate >= 0], g[g.replicate == -1])
            stab.append(rs.assign(target=t))
        stab = self.save(pd.concat(stab), "sobol_rank_stability")
        summ = (allidx[allidx.replicate >= 0].groupby(["target", "parameter"])
                .agg(ST_mean=("ST", "mean"), ST_q025=("ST", lambda z: z.quantile(.025)),
                     ST_q975=("ST", lambda z: z.quantile(.975)), S1_mean=("S1", "mean"),
                     S1_q025=("S1", lambda z: z.quantile(.025)),
                     S1_q975=("S1", lambda z: z.quantile(.975))).reset_index())
        summ = summ.merge(allidx[allidx.replicate == -1][["target", "parameter", "S1", "ST"]]
                          .rename(columns=dict(S1="S1_frozen", ST="ST_frozen")), on=["target", "parameter"])
        self.save(summ, "sobol_surrogate_summary")
        shap_rows, prov = [], {}
        for t in TARGETS:
            values, info = sa.ensemble_shap(self.surrogate, t, X)
            prov[t] = info
            sv = pd.DataFrame(values, columns=FEATURES).assign(target=t, setting_id=train.setting_id)
            shap_rows.append(sv)
        shap_df = self.save(pd.concat(shap_rows, ignore_index=True), "shap_values")
        share = (shap_df.groupby("target")[FEATURES].apply(lambda d: d.abs().mean() / d.abs().mean().sum())
                 .reset_index().melt(id_vars="target", var_name="parameter", value_name="mean_abs_SHAP_share"))
        self.save(share, "shap_shares")
        self.json("shap_provenance", prov)
        ale = pd.concat([sa.ale_with_refits(self.surrogate, t, train, min(c["n_boot"], 30)) for t in TARGETS])
        self.save(ale, "ale_refits")
        dml = pd.concat([sa.dml_projection(train, t, 40 if self.profile == "smoke" else 250) for t in TARGETS])
        self.save(dml, "dml_projection")
        return stab.groupby("target")[["kendall_tau", "top2_same"]].mean()

    # =====================================================================================
    # 11. Learning curve against the simulation-noise floor (development data only)
    # =====================================================================================
    def stage_learning_curve(self):
        from sklearn.base import clone
        from sklearn.model_selection import KFold
        train = self.data("train")
        sizes = [n for n in [16, 32, 64, 128, 256, 512] if n <= len(train)]
        rows = []
        for n in sizes:
            sub = train.iloc[:n]  # Sobol prefixes keep their balance
            for fold, (a, b) in enumerate(KFold(5, shuffle=True, random_state=826).split(sub)):
                for t in TARGETS:
                    preds = {}
                    for m in ["Ridge", "GaussianProcess", "CatBoost", "XGBoost"]:
                        mod = clone(self.surrogate.fitted[t][m]).fit(sub.iloc[a][FEATURES], sub.iloc[a][t])
                        preds[m] = mod.predict(sub.iloc[b][FEATURES])
                    preds["Ensemble"] = 0.5 * (preds["CatBoost"] + preds["XGBoost"])
                    for m, p in preds.items():
                        rows.append(dict(n_train=n, fold=fold, target=t, model=m,
                                         **metrics(sub.iloc[b][t], p)))
        return self.save(pd.DataFrame(rows), "learning_curve")

    # =====================================================================================
    # 12. Teleport rule sensitivity (identical settings and seeds, teleporting disabled)
    # =====================================================================================
    def stage_teleport(self):
        train = self.data("train").sort_values(TARGETS[0]).reset_index(drop=True)
        ix = np.linspace(0, len(train) - 1, self.cfg["n_teleport"]).round().astype(int)
        sel = train.iloc[ix]
        sim_off = Simulator(self.cache, teleport=-1, workers=self.workers)
        jobs = [dict(params=dz.params(r), seed=int(r.seed), meta=dict(setting_id=r.setting_id))
                for _, r in sel.iterrows()]
        off = self.simulate(jobs, "teleport_disabled_runs", sim=sim_off)
        keep = ["setting_id", "mean_time_loss_sec", "total_vkt_km", "total_completed_trips",
                "unfinished_trips", "stat_teleports_total"]
        pairs = sel[keep].merge(off[keep], on="setting_id", suffixes=("_tp300", "_off"))
        return self.save(pairs, "teleport_pairs")

    # =====================================================================================
    # 13. Static vs adaptive rerouting and network state at one representative setting
    # =====================================================================================
    def stage_routing(self):
        import gzip, shutil
        import xml.etree.ElementTree as ET
        from .simulator import _checked
        train = self.data("train").sort_values(TARGETS[0]).reset_index(drop=True)
        row = train.iloc[len(train) // 2]
        src = self.sim.root / "runs" / row.run_key / "routes.rou.xml.gz"
        d = self.root / "routing"
        d.mkdir(exist_ok=True)
        with gzip.open(src, "rb") as a, open(d / "routes.rou.xml", "wb") as b:
            shutil.copyfileobj(a, b)
        geom = self.sim.geometry()
        atomic_json(self.root / "tables" / "network_geometry.json", geom)
        rows, spatial = [], []
        p = self.sim.physics
        for mode, prob in [("static", 0), ("adaptive", 1)]:
            ef, add = d / f"{mode}_edges.xml", d / f"{mode}_add.xml"
            add.write_text(f'<additional><edgeData id="e" file="{ef}" period="300" '
                           f'begin="{p["begin"]}" end="{p["end"]}"/></additional>')
            _checked(["sumo", "--net-file", self.sim.net_file, "--route-files", d / "routes.rou.xml",
                      "--additional-files", add, "--begin", p["begin"], "--end", p["end"],
                      "--time-to-teleport", p["teleport"], "--device.rerouting.probability", prob,
                      "--device.rerouting.period", 180, "--seed", int(row.seed),
                      "--no-step-log", "true"], d, f"{mode}.log")
            for it in ET.parse(ef).getroot().iter("interval"):
                cbd = []
                for e in it.iter("edge"):
                    if e.attrib["id"] not in geom:
                        continue
                    r = dict(mode=mode, begin=float(it.attrib["begin"]), edge=e.attrib["id"],
                             density=float(e.attrib.get("density", 0)),
                             speed_kmh=float(e.attrib.get("speed", p["speed"])) * 3.6,
                             occupancy=float(e.attrib.get("occupancy", 0)))
                    spatial.append(r)
                    if geom[r["edge"]]["zone"] == "employment":
                        cbd.append(r)
                if cbd:
                    a = pd.DataFrame(cbd)
                    rows.append(dict(mode=mode, time_h=(float(it.attrib["begin"]) + 150) / 3600,
                                     density=a.density.mean(), flow=(a.density * a.speed_kmh).mean(),
                                     speed=a.speed_kmh.mean()))
        self.save(pd.DataFrame(spatial), "routing_edge_states")
        self.json("routing_provenance", dict(setting_id=row.setting_id, seed=int(row.seed),
                                             selection="median development delay"))
        return self.save(pd.DataFrame(rows), "routing_mfd")

    # =====================================================================================
    # 14. Transfer to an 8x8 grid: frozen surrogates, then prespecified adaptation
    # =====================================================================================
    def stage_transfer(self):
        from .adaptation import adapt
        c = self.cfg
        sim8 = Simulator(self.cache, grid_number=8, workers=self.workers)
        atomic_json(self.root / "tables" / "transfer_network.json", read_json(sim8.root / "simulator.json"))
        parts = dict(transfer=dz.random_design(c["n_transfer"], DESIGN_SEEDS["transfer"]),
                     adapt=dz.sobol_design(c["n_adapt"], DESIGN_SEEDS["adapt"]),
                     adapt_calibration=dz.random_design(c["n_adapt_cal"], DESIGN_SEEDS["adapt_calibration"]),
                     adapt_test=dz.random_design(c["n_adapt_test"], DESIGN_SEEDS["adapt_test"]))
        dz.assert_disjoint(parts)
        runs = {k: self.simulate(dz.jobs(v, RUN_SEED_BASE[k], k), f"{k}_runs_grid8", sim=sim8)
                for k, v in parts.items()}
        m, p, _ = val.independent_test(self.surrogate, runs["transfer"], TARGETS)
        self.save(m[m.estimand == "single_run"], "transfer_frozen_metrics")
        self.save(p, "transfer_frozen_predictions")
        methods = joblib.load(self.root / "models" / "interval_methods.joblib")
        cov, _, _ = val.evaluate_intervals(methods, self.data("train"), runs["transfer"], TARGETS)
        self.save(cov[cov.estimand == "single_run"], "transfer_frozen_interval_coverage")
        res = adapt(self.surrogate, runs["adapt"], runs["adapt_calibration"], runs["adapt_test"],
                    TARGETS, c["model_size"])
        for k, v in res.items():
            self.save(v, f"adaptation_{k}")
        return res["test_metrics"]

    # =====================================================================================
    # 15. Runtime and simulation-quality audit over every simulation in the study
    # =====================================================================================
    def stage_runtime_quality(self):
        frames = []
        for f in sorted((self.root / "tables").glob("*.csv")):
            if f.stem in ("train", "calibration", "test") or f.stem.endswith("_runs") or f.stem.endswith("_runs_grid8"):
                d = pd.read_csv(f)
                if "run_key" in d:
                    frames.append(d.assign(experiment=f.stem))
        allruns = pd.concat(frames, ignore_index=True)
        cols = ["experiment", "setting_id", "seed", "run_key", "elapsed_sec", "mean_time_loss_sec",
                "total_completed_trips", "unfinished_trips", "completion_fraction_recorded",
                "stat_teleports_total", "stat_vehicles_waiting", "stat_safety_collisions"]
        audit = allruns[[c for c in cols if c in allruns]]
        self.save(audit, "simulation_audit")
        exp = (audit.groupby("experiment").agg(runs=("run_key", "count"), unique_runs=("run_key", "nunique"),
                                               mean_sec=("elapsed_sec", "mean"), total_min=("elapsed_sec", lambda s: s.sum() / 60),
                                               runs_with_teleports=("stat_teleports_total", lambda s: int((s > 0).sum())),
                                               max_teleports=("stat_teleports_total", "max"),
                                               min_completion=("completion_fraction_recorded", "min"))
               .reset_index())
        self.save(exp, "simulation_audit_by_experiment")
        # surrogate evaluation cost
        X = dz.random_design(10_000, 5)
        t0 = time.perf_counter()
        self.surrogate.predict(X, TARGETS[0])
        per = (time.perf_counter() - t0) / len(X)
        unique = audit.drop_duplicates("run_key")
        self.json("runtime", dict(total_simulations=int(len(unique)),
                                  sumo_pipeline_mean_sec=float(unique.elapsed_sec.mean()),
                                  sumo_pipeline_median_sec=float(unique.elapsed_sec.median()),
                                  total_cpu_hours=float(unique.elapsed_sec.sum() / 3600),
                                  surrogate_sec_per_prediction=per,
                                  speedup=float(unique.elapsed_sec.mean() / per),
                                  workers=self.workers, machine=platform.processor() or platform.machine()))
        return exp

    # =====================================================================================
    def stage_figures(self):
        from . import figures
        return figures.make_all(self)

    def stage_export(self):
        from .export import export
        return export(self)


def objective_label(o):
    return {"aggregate": "Aggregate (2 outputs)", "extended": "Extended (8 outputs)"}.get(o, o)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the SUMO surrogate study")
    ap.add_argument("--output", default="results")
    ap.add_argument("--profile", choices=list(PROFILES), default="smoke")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    a = ap.parse_args()
    s = Study(a.output, a.profile, a.workers)
    s.run_all() if a.stage == "all" else s.run(a.stage)


if __name__ == "__main__":
    main()

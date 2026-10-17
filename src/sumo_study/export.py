"""Freeze the results: key numbers, LaTeX tables, artifact hashes and a review archive."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from .config import TARGETS
from .io import atomic_json, file_sha256, save_csv

NAMES = {"mean_time_loss_sec": "All-vehicle delay", "total_vkt_km": "VKT",
         "general_time_loss_sec": "General-traffic delay"}


def _tex(df: pd.DataFrame, path, caption, label, fmt="%.3f"):
    df = df.copy()
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype(str).str.replace("_", r"\_", regex=False).str.replace("±", r"$\pm$", regex=False)
    df.columns = [c.replace("_", r"\_") for c in df.columns]
    body = df.to_latex(index=False, float_format=lambda v: fmt % v, escape=False)
    path.write_text("\\begin{table}[!htbp]\n\\centering\n\\small\n"
                    f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n")


def key_numbers(S) -> dict:
    L = S.load
    out = {}
    tm = L("test_metrics")
    out["test"] = {f"{t}|{m}|{e}": dict(R2=r.R2, RMSE=r.RMSE, MAE=r.MAE, bias=r.bias)
                   for (t, m, e), r in tm.set_index(["target", "model", "estimand"]).iterrows()}
    cv = L("dev_cv_summary")
    out["cv"] = {f"{r.target}|{r.model}": dict(R2=r.R2_mean, R2_SD=r.R2_SD, RMSE=r.RMSE_mean)
                 for r in cv.itertuples()}
    out["noise"] = pd.concat([L("noise_replication"), L("noise_test_settings")]).to_dict("records")
    out["intervals"] = L("interval_coverage").to_dict("records")
    out["recourse_overview"] = L("recourse_overview").to_dict("records")
    out["mechanism_shares"] = L("mechanism_variance_shares").to_dict("records")
    c = L("mechanism_headway_contrasts")
    out["mechanism_contrasts_mean"] = (c.groupby(["background", "outcome"])
                                       [["mean_difference", "CI_low", "CI_high"]].mean()
                                       .reset_index().to_dict("records"))
    out["sobol_sumo"] = L("sobol_sumo_indices").to_dict("records")
    out["sobol_surrogate"] = L("sobol_surrogate_summary").to_dict("records")
    st = L("sobol_rank_stability")
    out["rank_stability"] = st.groupby("target").agg(tau=("kendall_tau", "median"),
                                                     top2=("top2_same", "mean")).reset_index().to_dict("records")
    out["shap_shares"] = L("shap_shares").to_dict("records")
    out["dml"] = L("dml_projection").to_dict("records")
    rec = L("inverse_parameter_recovery")
    out["inverse_recovery"] = rec.groupby(["objective", "parameter"]).abs_error.median().reset_index().to_dict("records")
    out["inverse_validation"] = (L("inverse_validation").groupby(["objective", "metric", "in_objective"])
                                 .relative_error_pct.median().reset_index().to_dict("records"))
    out["equifinality"] = L("inverse_equifinality").to_dict("records")
    out["transfer"] = L("transfer_frozen_metrics").to_dict("records")
    out["transfer_intervals"] = L("transfer_frozen_interval_coverage").to_dict("records")
    out["adaptation"] = L("adaptation_test_metrics").to_dict("records")
    out["adaptation_intervals"] = L("adaptation_intervals").to_dict("records")
    tel = L("teleport_pairs")
    out["teleport"] = dict(n=len(tel), max_abs_delay_diff=float((tel.mean_time_loss_sec_off - tel.mean_time_loss_sec_tp300).abs().max()),
                           teleports_total=float(tel.stat_teleports_total_tp300.sum()))
    out["runtime"] = json.loads((S.root / "tables" / "runtime.json").read_text())
    out["audit"] = L("simulation_audit_by_experiment").to_dict("records")
    return out


def latex_tables(S):
    d = S.root / "latex_tables"
    d.mkdir(exist_ok=True)
    cv = S.load("dev_cv_summary")
    tm = S.load("test_metrics")
    rows = []
    for t in TARGETS:
        for m in ["Ridge", "GaussianProcess", "RandomForest", "LightGBM", "CatBoost", "XGBoost", "Ensemble"]:
            c = cv[(cv.target == t) & (cv.model == m)].iloc[0]
            s = tm[(tm.target == t) & (tm.model == m) & (tm.estimand == "single_run")].iloc[0]
            rows.append({"Output": NAMES[t], "Model": m, "CV $R^2$": f"{c.R2_mean:.3f} ± {c.R2_SD:.3f}",
                         "CV RMSE": c.RMSE_mean, "Test $R^2$": s.R2, "Test RMSE": s.RMSE, "Test MAE": s.MAE})
    _tex(pd.DataFrame(rows), d / "tab_surrogate_performance.tex",
         "Development cross-validation and independent single-run test performance.", "tab:performance")
    cov = S.load("interval_coverage")
    cov = cov[cov.estimand != "seed_mean"].assign(Output=lambda x: x.target.map(NAMES))
    _tex(cov[["Output", "method", "estimand", "n", "coverage", "CI_low", "CI_high", "mean_width",
              "interval_score"]], d / "tab_interval_coverage.tex",
         "Independent-test coverage of 90\\% prediction intervals.", "tab:intervals")
    rv = S.load("recourse_validation")
    _tex(rv[rv.arm == "recourse"][["setting_id", "predicted_reduction_pct", "achieved_reduction_pct",
                                    "CI_low", "CI_high", "general_reduction_pct", "completion_ratio",
                                    "joint_met"]], d / "tab_recourse.tex",
         "Surrogate-predicted and SUMO-achieved delay reductions (10 matched seeds).", "tab:recourse", "%.2f")
    sob = S.load("sobol_sumo_indices")
    _tex(sob[sob.estimator != "SUMO raw"][["target", "estimator", "parameter", "S1", "ST", "ST_low", "ST_high"]],
         d / "tab_sobol.tex", "Direct-SUMO and surrogate Sobol indices.", "tab:sobol")
    return sorted(p.name for p in d.iterdir())


def export(S):
    atomic_json(S.root / "key_numbers.json", key_numbers(S))
    latex_tables(S)
    files = []
    for folder in ["tables", "figures", "designs", "latex_tables"]:
        for f in sorted((S.root / folder).rglob("*")):
            if f.is_file():
                files.append(dict(path=str(f.relative_to(S.root)), bytes=f.stat().st_size,
                                  sha256=file_sha256(f)))
    save_csv(pd.DataFrame(files), S.root / "artifact_manifest.csv")
    dest = S.root.parent / f"{S.profile}_results_archive.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(S.root / f["path"], f["path"])
        for n in ["manifest.json", "stage_status.json", "key_numbers.json", "artifact_manifest.csv"]:
            z.write(S.root / n, n)
        z.write(S.root / "models" / "freeze_record.json", "models/freeze_record.json")
    print("archive:", dest)
    return dest

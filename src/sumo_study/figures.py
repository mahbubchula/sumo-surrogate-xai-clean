"""Manuscript figures (figure-studio 'classic' theme, 180 mm, upper/lower panel layout).

Every figure is drawn from saved result tables only; nothing is refitted or re-simulated.
Run ``python -m sumo_study.figures results/full`` or the ``figures`` workflow stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import fstudio as F
from . import plots as P
from .config import FEATURES, LABELS, SHORT, TARGET_LABELS, TARGETS

F.use_theme("classic")
MODELS = ["Ridge", "GaussianProcess", "RandomForest", "LightGBM", "CatBoost", "XGBoost"]
MODEL_LABEL = {"Ridge": "Ridge", "GaussianProcess": "GP", "RandomForest": "RF",
               "LightGBM": "LightGBM", "CatBoost": "CatBoost", "XGBoost": "XGBoost",
               "Ensemble": "Ensemble"}
MCOL = dict(zip(MODELS, F.C6), Ensemble=F.INK)
TCOL = dict(zip(TARGETS, [F.S4[1], F.S4[0], F.S4[2]]))
TSHORT = {"mean_time_loss_sec": "All-vehicle delay [s]", "total_vkt_km": "VKT [km]",
          "general_time_loss_sec": "General-traffic delay [s]"}
METHOD_COL = {"CQR": F.S4[0], "SplitCP": F.S4[1], "GP": F.S4[3], "GP-CP": F.S4[2]}
DELAYS = ["mean_time_loss_sec", "general_time_loss_sec"]


class Tables:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, name):
        return pd.read_csv(self.root / "tables" / f"{name}.csv")

    def json(self, name):
        return json.loads((self.root / "tables" / f"{name}.json").read_text())


def _diag(ax, lo, hi):
    ax.plot([lo, hi], [lo, hi], color=F.INK2, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _lims(*arrs, pad=0.04):
    v = np.concatenate([np.asarray(a, float) for a in arrs])
    lo, hi = np.nanmin(v), np.nanmax(v)
    d = (hi - lo) * pad
    return lo - d, hi + d



# =========================================================================================
def fig00_framework(T, out):
    """Methodological framework: simulation model, partitions, and claim-versus-check design."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    aud = T("simulation_audit_by_experiment").set_index("experiment").runs
    W, H = 180, 162
    fig = F.new_figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    TEAL, BLUE, RED, AMB, PUR, GRY = F.S4[2], F.G0, F.G1, F.S4[1], F.S4[3], "#5B6770"
    R = 1.4  # corner radius [mm]

    def card(x, y, w, h, title, color, lines=(), hh=5.4, fs=6.8, body_fill=None, big=None, bigsize=13):
        """White card with a coloured title band."""
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={R}",
                                    fc=body_fill or "white", ec=color, lw=0.8, zorder=2))
        ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle=f"round,pad=0,rounding_size={R}",
                                    fc=color, ec=color, lw=0.8, zorder=3))
        ax.add_patch(Rectangle((x, y + h - hh), w, hh / 2, fc=color, ec="none", zorder=3))
        ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center", fontsize=7.3,
                fontweight="bold", color="white", zorder=4)
        top = y + h - hh - 2.2
        if big:
            ax.text(x + w / 2, top - 0.4, big, ha="center", va="top", fontsize=bigsize, fontweight="bold",
                    color=color, zorder=4)
            top -= 7.4
        for i, line in enumerate(lines):
            ax.text(x + w / 2, top - 3.45 * i, line, ha="center", va="top", fontsize=fs, color=F.INK, zorder=4)

    def arrow(x0, y0, x1, y1, color=GRY, lw=1.1, ms=9):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color,
                                     lw=lw, shrinkA=0, shrinkB=0, zorder=5))

    def header(y, letter, text):
        ax.text(1.5, y, f"({letter})", fontsize=10, fontweight="bold", va="center", color=F.INK)
        ax.text(9.5, y, text, fontsize=9.2, fontweight="bold", va="center", color=F.INK)
        ax.plot([9.5 + 0.0, W - 1.5], [y - 3.0, y - 3.0], color=F.RULE, lw=0.6)

    # ------------------------------------------------------------- (a) simulation model
    header(158.5, "a", "Stochastic simulation model")
    ya, ha = 127.5, 26
    tiles = [(1.5, 38, "Inputs θ (uniform)", ["Car ownership, car preference", "Unemployment, free-time rate",
                                                   "Departure variation", "Walking limit, bus headway",
                                                   "Random traffic, peak share"]),
             (45.5, 38, "Demand generation", ["SUMO activitygen", "3,500 inhabitants, households,",
                                             "work, school, free time", "Trips of cars, random", "traffic and buses"]),
             (89.5, 41, "Routing and simulation", ["duarouter (static routes)", "sumo, 06:30–10:30",
                                                   "6×6 signalized grid, 120 edges", "One seed ω for all stages",
                                                   "Runs cached by (θ, ω)"]),
             (136.5, 42, "Outputs Y(θ, ω)", ["All-vehicle delay", "General-traffic delay",
                                                       "(buses excluded)", "Vehicle-kilometers traveled",
                                                       "Trips by class, hourly arrivals"])]
    for x, w, t, ls in tiles:
        card(x, ya, w, ha, t, TEAL, ls)
    for (x0, w0, *_), (x1, *_) in zip(tiles[:-1], tiles[1:]):
        arrow(x0 + w0 + 0.6, ya + ha / 2 - 2, x1 - 0.6, ya + ha / 2 - 2, color=TEAL)

    # ------------------------------------------------------------- (b) partitions
    header(119.5, "b", "Pre-specified partitions with disjoint settings and seeds")
    yb, hb = 90, 25
    parts = [(1.5, "Development", BLUE, "512", ["Scrambled Sobol, 1 seed", "Fit and cross-validate"]),
             (46.5, "Calibration", AMB, "200", ["Uniform random, 1 seed", "Calibrate intervals only"]),
             (91.5, "Test", PUR, "200 × 5", ["Uniform random, 5 seeds", "Evaluate once, untouched"])]
    for x, t, c, big, ls in parts:
        card(x, yb, 43, hb, t, c, ls, big=big, body_fill=F.tint(c, 0.94))
    card(136.5, yb, 42, hb, "Analysis protocol", GRY,
         ["Sizes, seeds, models and", "decision rules fixed", "before the full run;", "models frozen and hashed",
          "before any test outcome"], body_fill=F.tint(GRY, 0.93))

    # ------------------------------------------------------------- (c) claims and checks
    header(83.5, "c", "Each surrogate claim is checked by a dedicated SUMO experiment")
    yc0, yc1 = 1.5, 76
    # surrogate card
    xs, ws = 1.5, 34
    ax.add_patch(FancyBboxPatch((xs, yc0), ws, yc1 - yc0, boxstyle=f"round,pad=0,rounding_size={R}",
                                fc=F.tint(BLUE, 0.94), ec=BLUE, lw=0.8, zorder=2))
    ax.add_patch(FancyBboxPatch((xs, yc1 - 5.4), ws, 5.4, boxstyle=f"round,pad=0,rounding_size={R}",
                                fc=BLUE, ec=BLUE, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((xs, yc1 - 5.4), ws, 2.7, fc=BLUE, ec="none", zorder=3))
    ax.text(xs + ws / 2, yc1 - 2.7, "Surrogates", ha="center", va="center", fontsize=7.3, fontweight="bold",
            color="white", zorder=4)
    ax.text(xs + ws / 2, yc1 - 8.3, "Fitted on development data,", ha="center", va="top", fontsize=6.6, zorder=4)
    ax.text(xs + ws / 2, yc1 - 11.6, "fixed hyperparameters", ha="center", va="top", fontsize=6.6, zorder=4)
    names = ["Ridge", "GaussianProcess", "RandomForest", "LightGBM", "CatBoost", "XGBoost", "Ensemble"]
    labels = ["Ridge", "Gaussian process", "Random forest", "LightGBM", "CatBoost", "XGBoost", "Ensemble (primary)"]
    for i, (n, l) in enumerate(zip(names, labels)):
        yy = yc1 - 19.5 - 4.3 * i
        ax.add_patch(Circle((xs + 5.2, yy), 1.05, fc=MCOL[n], ec="none", zorder=4))
        ax.text(xs + 8.0, yy, l, va="center", fontsize=6.9, zorder=4,
                fontweight="bold" if n == "Ensemble" else "normal")
    ax.plot([xs + 3, xs + ws - 3], [yc1 - 50.5, yc1 - 50.5], color=F.tint(BLUE, 0.6), lw=0.6, zorder=4)
    ax.text(xs + ws / 2, yc1 - 53.0, "Prediction intervals", ha="center", va="top", fontsize=6.9,
            fontweight="bold", zorder=4)
    for i, l in enumerate(["CQR, split conformal,", "Gaussian process,", "GP-normalized conformal"]):
        ax.text(xs + ws / 2, yc1 - 57.0 - 3.4 * i, l, ha="center", va="top", fontsize=6.6, zorder=4)

    rows = [
        ("Predictions are accurate", "Frozen models on the untouched test set;", "seed-noise ceiling from replicates",
         aud.get("test", 0) + aud.get("replication_runs", 0)),
        ("90% intervals cover", "Coverage per run and per congestion", "tertile on the test set", None),
        ("Sensitivity rankings hold", "Direct-SUMO Sobol indices, noise-", "corrected; 50 surrogate refits",
         aud.get("sobol_sumo_runs", 0)),
        ("Bus headway reduces delay", "8×8 factorial, 10 common seeds,", "outcomes split by vehicle class",
         aud.get("mechanism_runs", 0)),
        ("Interventions cut delay 15%", "Matched seeds against baseline and", "a random change of equal size",
         aud.get("recourse_runs", 0)),
        ("Calibration recovers θ", "Excluded targets, fresh seeds,", "outputs outside the objective",
         aud.get("inverse_target_runs", 0) + aud.get("inverse_validation_runs", 0)),
        ("Models transfer", "8×8 grid: frozen transfer and", "four adaptation methods",
         sum(aud.get(k, 0) for k in ["transfer_runs_grid8", "adapt_runs_grid8",
                                    "adapt_calibration_runs_grid8", "adapt_test_runs_grid8"])),
    ]
    xc, wc, xk, wk = 41, 47, 95, 68
    ax.text(xc + wc / 2, yc1 - 1.6, "Surrogate claim", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=BLUE)
    ax.text(xk + wk / 2, yc1 - 1.6, "Simulator check in SUMO", ha="center", va="center", fontsize=7.6,
            fontweight="bold", color=RED)
    ax.text(171.5, yc1 - 1.6, "Runs", ha="center", va="center", fontsize=7.6, fontweight="bold", color=F.INK2)
    rh, gap = 8.4, 1.55
    top = yc1 - 5.2
    for i, (claim, c1, c2, n) in enumerate(rows):
        y = top - (i + 1) * rh - i * gap
        ym = y + rh / 2
        ax.add_patch(FancyBboxPatch((xc, y), wc, rh, boxstyle=f"round,pad=0,rounding_size={R}",
                                    fc="white", ec=BLUE, lw=0.8, zorder=2))
        ax.add_patch(Circle((xc + 4.6, ym), 2.55, fc=BLUE, ec="none", zorder=4))
        ax.text(xc + 4.6, ym, str(i + 1), ha="center", va="center", fontsize=7.2, fontweight="bold",
                color="white", zorder=5)
        ax.text(xc + 9.2, ym, claim, va="center", fontsize=7.0, color=F.INK, zorder=4)
        ax.add_patch(FancyBboxPatch((xk, y), wk, rh, boxstyle=f"round,pad=0,rounding_size={R}",
                                    fc=F.tint(RED, 0.92), ec=RED, lw=0.8, zorder=2))
        ax.text(xk + 3, ym + 1.75, c1, va="center", fontsize=6.8, color=F.INK, zorder=4)
        ax.text(xk + 3, ym - 1.75, c2, va="center", fontsize=6.8, color=F.INK, zorder=4)
        if n:
            ax.add_patch(FancyBboxPatch((165.5, ym - 2.4), 12.5, 4.8, boxstyle="round,pad=0,rounding_size=2.4",
                                        fc=F.tint(GRY, 0.85), ec="none", zorder=2))
            ax.text(171.75, ym, f"{int(n):,}", ha="center", va="center", fontsize=6.9, fontweight="bold",
                    color=F.INK, zorder=4)
        else:
            ax.text(171.75, ym, "test set", ha="center", va="center", fontsize=6.4, color=F.INK2, zorder=4)
        arrow(xs + ws + 0.4, ym, xc - 0.4, ym, color=BLUE, lw=0.8, ms=7)
        arrow(xc + wc + 0.4, ym, xk - 0.4, ym, color=RED, lw=0.9, ms=7)
    F.save(fig, out / "Fig01_methodological_framework")


# =========================================================================================
def fig01_design(T, out):
    geom = T.json("network_geometry")
    aud = T("simulation_audit_by_experiment")
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    g0 = gs[0].subgridspec(1, 2, width_ratios=[1, 1.25])
    g1 = gs[1].subgridspec(1, 2)
    ax = fig.add_subplot(g0[0])
    segs, cols, lws = [], [], []
    for e in geom.values():
        segs.append(e["points"])
        cols.append(F.G1 if e["zone"] == "employment" else F.G0_LIGHT)
        lws.append(2.4 if e["bus"] else 1.1)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round"))
    for e in geom.values():
        if e["bus"]:
            p = np.array(e["points"])
            ax.plot(p[:, 0], p[:, 1], color=F.S4[2], lw=2.6, solid_capstyle="round", zorder=3)
        if e["gate"] or e["school"]:
            p = np.array(e["points"]).mean(0)
            ax.scatter(*p, marker="s" if e["gate"] else "^", s=34, color=F.INK, ec=F.BG, zorder=4)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color=F.G1, lw=2, label="Employment zone"),
                       Line2D([], [], color=F.G0_LIGHT, lw=2, label="Residential zone"),
                       Line2D([], [], color=F.S4[2], lw=2.6, label="Bus line edges"),
                       Line2D([], [], marker="s", ls="", color=F.INK, label="City gate"),
                       Line2D([], [], marker="^", ls="", color=F.INK, label="School")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7.5)
    F.panel(fig, ax, "a", "Grid network")
    ax = fig.add_subplot(g0[1])
    names = {"train": "Development (Sobol)", "calibration": "Interval calibration", "test": "Test (5 seeds)",
             "replication_runs": "Seed replication", "recourse_runs": "Recourse (3 arms)",
             "inverse_target_runs": "Inverse-calib. targets", "inverse_validation_runs": "Inverse-calib. check",
             "mechanism_runs": "Mechanism factorial", "sobol_sumo_runs": "Direct-SUMO Sobol",
             "teleport_disabled_runs": "Teleport off", "transfer_runs_grid8": "8×8 transfer",
             "adapt_runs_grid8": "8×8 adaptation", "adapt_calibration_runs_grid8": "8×8 interval calib.",
             "adapt_test_runs_grid8": "8×8 test"}
    a = aud[aud.experiment.isin(names)].copy()
    a["label"] = a.experiment.map(names)
    a = a.sort_values("runs")
    y = np.arange(len(a))
    ax.barh(y, a.runs, color=F.G0, height=0.62)
    for yy, v in zip(y, a.runs):
        ax.text(v, yy, f" {v:,}", va="center", fontsize=7.5)
    ax.set_yticks(y, a.label)
    F.plain_log(ax, [10, 100, 1000, 10000])
    ax.set_xlim(10, a.runs.max() * 6)
    ax.set_xlabel("SUMO runs (log scale)")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "b", f"Run budget, {int(aud.runs.sum()):,} runs")
    parts = {k: T(k) for k in ["train", "calibration", "test"]}
    parts["test"] = parts["test"][parts["test"].replicate == 0]
    for j, (t, letter) in enumerate(zip(DELAYS, "cd")):
        ax = fig.add_subplot(g1[j])
        P.raincloud(ax, [parts[k][t].to_numpy() for k in ["test", "calibration", "train"]],
                    ["Test", "Calib.", "Develop."], colors=[F.S4[2], F.S4[1], F.S4[0]],
                    show_stats=False)
        ax.set_xlabel(TSHORT[t])
        F.panel(fig, ax, letter, "All-vehicle delay" if j == 0 else "General-traffic delay")
    F.save(fig, out / "Fig02_design_and_network")


# =========================================================================================
def fig02_noise(T, out):
    runs = T("replication_runs")
    noise = pd.concat([T("noise_replication"), T("noise_test_settings")])
    fig = F.new_figure(180, 140)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1])
    g0 = gs[0].subgridspec(1, 2)
    for j, (t, letter) in enumerate(zip(DELAYS, "ab")):
        ax = fig.add_subplot(g0[j])
        m = runs.groupby("setting_id")[t].agg(["mean", "std", "min", "max"]).sort_values("mean")
        lvl = runs.groupby("setting_id").congestion_level.first().reindex(m.index)
        x = np.arange(len(m))
        cmap = {"low": F.ORD[0], "medium": F.ORD[1], "high": F.ORD[3]}
        for i, sid in enumerate(m.index):
            v = runs.loc[runs.setting_id == sid, t]
            ax.scatter(np.full(len(v), i), v, s=9, color=F.tint(cmap[lvl[sid]], 0.35), lw=0, zorder=2)
        ax.errorbar(x, m["mean"], yerr=1.96 * m["std"] / np.sqrt(runs.groupby("setting_id").size().reindex(m.index)),
                    fmt="o", ms=3.5, color=F.INK, lw=1, capsize=0, zorder=3)
        ax.set_xticks([])
        ax.set_xlabel("Replicated settings, ordered by seed mean")
        ax.set_ylabel(TSHORT[t])
        F.ygrid(ax)
        if j == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=cmap[k], label=k.capitalize())
                               for k in cmap] + [Line2D([], [], marker="o", color=F.INK, label="Mean ±95% CI")],
                      loc="upper left", fontsize=7.5, title="Delay tertile", title_fontsize=7.5)
        F.panel(fig, ax, letter, "Seed spread, all vehicles" if j == 0 else "Seed spread, general traffic")
    g1 = gs[1].subgridspec(1, 2)
    ax = fig.add_subplot(g1[0])
    sub = noise[noise.target.isin(TARGETS)]
    srcs = list(sub.source.unique())
    w = 0.36
    for k, s in enumerate(srcs):
        d = sub[sub.source == s].set_index("target").reindex(TARGETS)
        ax.bar(np.arange(3) + (k - 0.5) * w, 100 * d.noise_share, w * 0.92,
               color=[F.G0, F.G1][k], label=s.capitalize())
        for i, v in enumerate(100 * d.noise_share):
            ax.text(i + (k - 0.5) * w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(range(3), ["All-vehicle\ndelay", "VKT", "General\ndelay"])
    ax.set_ylabel("Seed-noise share [%]")
    ax.set_ylim(0, 100 * sub.noise_share.max() * 1.45)
    ax.legend(fontsize=7.3, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Noise share")
    ax = fig.add_subplot(g1[1])
    tm = T("test_metrics")
    tm = tm[(tm.estimand == "single_run") & (tm.model == "Ensemble")].set_index("target")
    ts = T("noise_test_settings").set_index("target")
    names = ["All-vehicle delay", "VKT", "General delay"]
    y = np.arange(3)[::-1]
    for yy, t in zip(y, TARGETS):
        floor, rmse = ts.loc[t, "RMSE_floor_single_run"], tm.loc[t, "RMSE"]
        ax.barh(yy, rmse / rmse * 100, 0.6, color=F.tint(F.INK, 0.86))
        ax.barh(yy, floor / rmse * 100, 0.6, color=F.G0)
        u = "km" if "vkt" in t else "s"
        ax.text(floor / rmse * 100 - 2, yy, f"{floor:.1f} / {rmse:.1f} {u}", ha="right", va="center",
                fontsize=7.4, color="white", fontweight="bold")
        ax.text(101, yy, f"{floor / rmse:.0%}", va="center", fontsize=7.6)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Noise SD / ensemble test RMSE [%]")
    ax.legend(handles=[Patch(color=F.G0, label="Seed-noise SD (floor)"),
                       Patch(color=F.tint(F.INK, 0.86), label="Ensemble single-run RMSE")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.3)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "d", "Noise floor")
    F.save(fig, out / "Fig03_simulation_noise")


# =========================================================================================
def fig03_development(T, out):
    cv = T("dev_cv_summary")
    lc = T("learning_curve")
    noise = T("noise_replication").set_index("target")
    order = MODELS + ["Ensemble"]
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0])
    cols, data, hb = [], [], []
    for t in TARGETS:
        s = cv[cv.target == t].set_index("model").reindex(order)
        data += [s.R2_mean.to_numpy(), s.RMSE_mean.to_numpy()]
        u = "s" if "time" in t else "km"
        nm = {"mean_time_loss_sec": "All delay", "total_vkt_km": "VKT", "general_time_loss_sec": "Gen. delay"}[t]
        cols += [f"{nm} R²", f"{nm} RMSE"]
        hb += [True, False]
    P.heat_table(ax, np.column_stack(data), [MODEL_LABEL[m] for m in order], cols, higher_better=hb,
                 fmt="{:.2f}", row_colors=[MCOL[m] for m in order])
    F.panel(fig, ax, "a", "Development 5-fold CV")
    g1 = gs[1].subgridspec(1, 3)
    for j, t in enumerate(TARGETS):
        ax = fig.add_subplot(g1[j])
        d = lc[lc.target == t].groupby(["model", "n_train"]).RMSE.agg(["mean", "std"]).reset_index()
        for m in ["Ridge", "GaussianProcess", "CatBoost", "XGBoost", "Ensemble"]:
            g = d[d.model == m]
            ax.plot(g.n_train, g["mean"], "-o", ms=3, color=MCOL[m], label=MODEL_LABEL[m],
                    lw=2 if m == "Ensemble" else 1.3)
        fl = noise.loc[t, "RMSE_floor_single_run"]
        ax.axhline(fl, color=F.INK2, ls="--", lw=1)
        ax.text(16, fl, " seed-noise floor", va="bottom", fontsize=7.3, color=F.INK2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.n_train.unique()), [str(v) for v in sorted(d.n_train.unique())])
        ax.set_xlabel("Development settings")
        ax.set_ylabel(f"CV RMSE [{'s' if 'time' in t else 'km'}]")
        ax.set_ylim(bottom=0)
        F.ygrid(ax)
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")
        F.panel(fig, ax, "bcd"[j], {"mean_time_loss_sec": "All-vehicle delay", "total_vkt_km": "VKT",
                                    "general_time_loss_sec": "General delay"}[t])
    F.save(fig, out / "Fig04_development_and_learning")


# =========================================================================================
def fig04_test(T, out):
    p = T("test_predictions")
    tm = T("test_metrics")
    cv = T("dev_cv_summary")
    pair = T("test_paired_mae_differences")
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    g0 = gs[0].subgridspec(1, 3)
    for j, t in enumerate(TARGETS):
        ax = fig.add_subplot(g0[j])
        e = p[(p.target == t) & (p.model == "Ensemble")]
        r = p[(p.target == t) & (p.model == "Ridge")]
        lo, hi = _lims(e.observed, e.predicted, r.predicted)
        _diag(ax, lo, hi)
        ax.scatter(r.observed, r.predicted, s=8, color=F.tint(MCOL["Ridge"], 0.45), lw=0, label="Ridge")
        ax.scatter(e.observed, e.predicted, s=10, color=F.INK, lw=0, label="Ensemble")
        m = tm[(tm.target == t) & (tm.estimand == "single_run")].set_index("model")
        ax.text(0.04, 0.96, f"Ensemble R² {m.loc['Ensemble', 'R2']:.3f}\nRidge R² {m.loc['Ridge', 'R2']:.3f}",
                transform=ax.transAxes, va="top", fontsize=7.4)
        ax.set_xlabel({"mean_time_loss_sec": "SUMO all-vehicle delay [s]", "total_vkt_km": "SUMO VKT [km]",
                       "general_time_loss_sec": "SUMO general delay [s]"}[t])
        ax.set_ylabel("Predicted")
        F.ygrid(ax)
        if j == 2:
            ax.legend(loc="lower right", fontsize=7.3)
        F.panel(fig, ax, "abc"[j], ["All-vehicle delay", "VKT", "General delay"][j])
    g1 = gs[1].subgridspec(1, 2, width_ratios=[1.05, 1])
    ax = fig.add_subplot(g1[0])
    order = MODELS + ["Ensemble"]
    for j, t in enumerate(TARGETS):
        c = cv[cv.target == t].set_index("model").reindex(order).R2_mean
        s = tm[(tm.target == t) & (tm.estimand == "single_run")].set_index("model").reindex(order).R2
        x = np.arange(len(order)) + (j - 1) * 0.26
        ax.vlines(x, c, s, color=F.tint(TCOL[t], 0.3), lw=1.4)
        ax.scatter(x, c, s=16, marker="o", facecolor=F.BG, edgecolor=TCOL[t], lw=1.1, zorder=3)
        ax.scatter(x, s, s=18, marker="o", color=TCOL[t], zorder=3,
                   label=["All delay", "VKT", "General delay"][j])
    ax.set_xticks(range(len(order)), [MODEL_LABEL[m] for m in order])
    F.rotate(ax, 30)
    ax.set_ylabel("R²  (open: CV, filled: test)")
    ax.legend(fontsize=7.2, loc="center", bbox_to_anchor=(0.62, 0.42), ncol=1)
    F.ygrid(ax)
    F.panel(fig, ax, "d", "Development vs test")
    ax = fig.add_subplot(g1[1])
    mae = tm[tm.estimand == "single_run"].set_index(["target", "model"]).MAE
    rows = []
    for t in TARGETS:
        short = {"mean_time_loss_sec": "All delay", "total_vkt_km": "VKT", "general_time_loss_sec": "Gen. delay"}[t]
        for m, ref, c in [("Ensemble", "Ridge", F.INK), ("GaussianProcess", "Ensemble", MCOL["GaussianProcess"])]:
            r = pair[(pair.target == t) & (pair.model == m) & (pair.reference == ref)].iloc[0]
            k = 100 / mae[(t, ref)]
            rows.append((f"{short}: {MODEL_LABEL[m]} vs {MODEL_LABEL[ref]}", r.MAE_difference * k,
                         r.CI_low * k, r.CI_high * k, c))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, e, lo, hi, c) in zip(y, rows):
        ax.axhspan(yy - .5, yy + .5, color=F.tint(F.INK, 0.96) if yy % 2 else F.BG, lw=0, zorder=0)
        ax.hlines(yy, lo, hi, color=c, lw=1.6, zorder=2)
        ax.scatter(e, yy, s=30, marker="s", color=c, zorder=3)
        ax.text(1.02, yy, F.minus(f"{e:+.0f}%"), transform=ax.get_yaxis_transform(), va="center", fontsize=7.6)
    ax.axvline(0, color=F.INK2, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("Paired change in test MAE [% of reference]")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "e", "Paired error differences")
    F.save(fig, out / "Fig05_independent_test")


# =========================================================================================
def fig05_intervals(T, out):
    cov = T("interval_coverage")
    pr = T("interval_test_predictions")
    st = T("interval_coverage_by_stratum")
    fig = F.new_figure(180, 155)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    g0 = gs[0].subgridspec(1, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(g0[0])
    methods = ["CQR", "SplitCP", "GP", "GP-CP"]
    rows = []
    for t in TARGETS:
        for m in methods:
            rows.append((t, m))
    y = np.arange(len(rows))[::-1]
    for (t, m), yy in zip(rows, y):
        for est, mk, dy in [("single_run", "o", 0.14), ("new_runs", "D", -0.14)]:
            r = cov[(cov.target == t) & (cov.method == m) & (cov.estimand == est)].iloc[0]
            ax.hlines(yy + dy, r.CI_low, r.CI_high, color=METHOD_COL[m], lw=1.3)
            ax.scatter(r.coverage, yy + dy, marker=mk, s=20, color=METHOD_COL[m] if est == "single_run" else F.BG,
                       edgecolor=METHOD_COL[m], lw=1, zorder=3)
        if yy % 4 == 3:
            pass
    for k in range(1, 3):
        ax.axhline(len(rows) - 4 * k - 0.5, color=F.RULE, lw=0.6)
    ax.axvline(0.9, color=F.G1, lw=1, ls="--")
    ax.set_yticks(y, [f"{m}" for t, m in rows], fontsize=7.6)
    for k, t in enumerate(TARGETS):
        ax.text(1.005, len(rows) - 1 - 4 * k - 1.5, ["All delay", "VKT", "Gen. delay"][k],
                transform=ax.get_yaxis_transform(), rotation=270, va="center", fontsize=7.6,
                fontweight="bold")
    ax.set_xlim(0.5, 1.0)
    F.pct_axis(ax, "x")
    ax.set_xlabel("Test coverage (nominal 90%)")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=F.INK2, label="Single future run"),
                       Line2D([], [], marker="D", ls="", mfc=F.BG, color=F.INK2, label="All 5 seeds")],
              loc="lower left", fontsize=7.2)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Coverage")
    ax = fig.add_subplot(g0[1])
    for t, mk in zip(TARGETS, ["o", "s", "^"]):
        c = cov[(cov.target == t) & (cov.estimand == "single_run")]
        ref = c.mean_width.min()
        for _, r in c.iterrows():
            ax.scatter(r.mean_width / ref, r.coverage, marker=mk, s=34, color=METHOD_COL[r.method],
                       ec=F.BG, zorder=3)
    ax.axhline(0.9, color=F.G1, ls="--", lw=1)
    F.pct_axis(ax)
    ax.set_xlabel("Mean width relative to narrowest")
    ax.set_ylabel("Single-run coverage")
    ax.legend(handles=[Patch(color=METHOD_COL[m], label=m) for m in methods] +
              [Line2D([], [], marker=mk, ls="", color=F.INK2, label=l) for mk, l in
               zip(["o", "s", "^"], ["All delay", "VKT", "Gen. delay"])],
              fontsize=7, ncol=2, loc="lower right")
    F.ygrid(ax)
    F.panel(fig, ax, "b", "Coverage vs width")
    g1 = gs[1].subgridspec(1, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(g1[0])
    best = "SplitCP"
    d = pr[(pr.target == "general_time_loss_sec") & (pr.method == best)].sort_values("observed").reset_index(drop=True)
    x = np.arange(len(d))
    ax.fill_between(x, d.lower, d.upper, color=F.tint(METHOD_COL[best], 0.65), lw=0, step="mid",
                    label="90% interval")
    ok = d.covered.astype(bool)
    ax.scatter(x[ok], d.observed[ok], s=7, color=F.INK, lw=0, label="Covered", zorder=3)
    ax.scatter(x[~ok], d.observed[~ok], s=16, color=F.G1, marker="x", lw=1.1, label="Missed", zorder=4)
    ax.set_xlabel("Test settings, ordered by SUMO outcome")
    ax.set_ylabel(TSHORT["general_time_loss_sec"])
    ax.legend(fontsize=7.3, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Split-conformal intervals")
    ax = fig.add_subplot(g1[1])
    s = st[(st.method.isin(["CQR", "SplitCP", "GP-CP"]))]
    strata = ["low", "medium", "high"]
    for k, m in enumerate(["CQR", "SplitCP", "GP-CP"]):
        g = s[(s.method == m) & (s.target == "general_time_loss_sec")].set_index("stratum").reindex(strata)
        xx = np.arange(3) + (k - 1) * 0.22
        ax.vlines(xx, g.CI_low, g.CI_high, color=METHOD_COL[m], lw=1.3)
        ax.scatter(xx, g.coverage, s=24, color=METHOD_COL[m], label=m, zorder=3)
    ax.axhline(0.9, color=F.G1, ls="--", lw=1)
    ax.set_xticks(range(3), ["Low", "Medium", "High"])
    ax.set_xlabel("Congestion tertile (general delay)")
    ax.set_ylabel("Coverage")
    F.pct_axis(ax)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=7.2, loc="lower left")
    F.ygrid(ax)
    F.panel(fig, ax, "d", "Conditional coverage")
    F.save(fig, out / "Fig06_predictive_intervals")


# =========================================================================================
def fig06_sobol(T, out):
    sumo = T("sobol_sumo_indices")
    summ = T("sobol_surrogate_summary")
    stab = T("sobol_rank_stability")
    fig = F.new_figure(180, 165)
    gs = fig.add_gridspec(2, 2)
    est = [("SUMO noise-corrected", F.G1, "s", "Direct SUMO (noise-corrected)"),
           ("Surrogate, same design", F.G0, "o", "Ensemble, same design")]
    titles = {"mean_time_loss_sec": "All-vehicle delay", "general_time_loss_sec": "General-traffic delay",
              "total_vkt_km": "VKT"}
    for k, (t, cell) in enumerate(zip(["mean_time_loss_sec", "general_time_loss_sec", "total_vkt_km"],
                                      [gs[0, 0], gs[0, 1], gs[1, 0]])):
        ax = fig.add_subplot(cell)
        ref = sumo[(sumo.target == t) & (sumo.estimator == "SUMO noise-corrected")].set_index("parameter")
        order = ref.ST.sort_values().index.tolist()
        y = np.arange(len(order))
        for j, (e, c, mk, lab) in enumerate(est):
            d = sumo[(sumo.target == t) & (sumo.estimator == e)].set_index("parameter").reindex(order)
            yy = y + (0.17 if j == 0 else -0.17)
            ax.hlines(yy, d.ST_low, d.ST_high, color=c, lw=1.3)
            ax.scatter(d.ST, yy, marker=mk, s=22, color=c, zorder=3, label=lab)
        f = summ[summ.target == t].set_index("parameter").reindex(order)
        ax.hlines(y, f.ST_q025, f.ST_q975, color=F.INK, lw=3.2, alpha=0.35, zorder=2)
        ax.scatter(f.ST_frozen, y, marker="|", s=70, color=F.INK, zorder=4,
                   label="Ensemble, 2\u00b9\u00b9 design (bar: 50 refits)")
        ax.axvline(0, color=F.INK2, lw=0.7)
        ax.set_yticks(y, [SHORT[p] for p in order])
        ax.set_xlabel("Total-order index S$_T$")
        ax.set_xlim(-0.03, 1.1)
        F.clean_left(ax)
        F.ygrid(ax, "x")
        if k == 0:
            ax.legend(fontsize=6.9, loc="lower right")
        F.panel(fig, ax, "abc"[k], titles[t])
    ax = fig.add_subplot(gs[1, 1])
    mk = {"mean_time_loss_sec": "o", "general_time_loss_sec": "s", "total_vkt_km": "^"}
    for t in TARGETS:
        r = sumo[(sumo.target == t) & (sumo.estimator == "SUMO noise-corrected")].set_index("parameter")
        f = summ[summ.target == t].set_index("parameter").reindex(r.index)
        ax.errorbar(f.ST_frozen, r.ST, yerr=[r.ST - r.ST_low, r.ST_high - r.ST], fmt=mk[t], ms=4.5,
                    color=TCOL[t], ecolor=F.tint(TCOL[t], 0.4), lw=0.9, label=titles[t])
    hb = sumo[(sumo.target == "mean_time_loss_sec") & (sumo.estimator == "SUMO noise-corrected")].set_index("parameter").loc["busHeadway"]
    ax.annotate("Headway,\nall vehicles", xy=(summ[(summ.target == "mean_time_loss_sec") & (summ.parameter == "busHeadway")].ST_frozen.iloc[0], hb.ST),
                xytext=(0.32, 0.45), fontsize=7.2, arrowprops=dict(arrowstyle="->", color=F.INK2, lw=0.7))
    _diag(ax, -0.03, 1.1)
    st = stab.groupby("target").agg(tau=("kendall_tau", "median"), top2=("top2_same", "mean"))
    ax.text(0.97, 0.04, "50 bootstrap refits:\n" + "\n".join(
        f"{ {'mean_time_loss_sec': 'All delay', 'total_vkt_km': 'VKT', 'general_time_loss_sec': 'Gen. delay'}[t]}: "
        f"top-2 kept {st.loc[t, 'top2']:.0%}, median \u03c4 = {st.loc[t, 'tau']:.2f}" for t in TARGETS),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9)
    ax.set_xlabel("Surrogate S$_T$ (frozen ensemble)")
    ax.set_ylabel("Direct-SUMO S$_T$ (95% CI)")
    ax.legend(fontsize=7, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "d", "Simulator vs surrogate")
    F.save(fig, out / "Fig07_sobol_sensitivity")


# =========================================================================================
def fig07_ale(T, out):
    ale = T("ale_refits")
    summ = T("sobol_surrogate_summary")
    fig = F.new_figure(180, 150)
    gs = fig.add_gridspec(2, 1)
    v = ale[ale.target.isin(DELAYS) & ale.parameter.isin(["departureVariation", "busHeadway", "footDistanceLimit"])].ALE
    pad = 0.06 * (v.max() - v.min())
    ylims = {k: (v.min() - pad, v.max() + pad) for k in ["departureVariation", "busHeadway", "footDistanceLimit"]}
    for row, t in enumerate(DELAYS):
        top = ["departureVariation", "busHeadway", "footDistanceLimit"]
        g = gs[row].subgridspec(1, 3)
        for j, k in enumerate(top):
            d = ale[(ale.target == t) & (ale.parameter == k)]
            base = d[d.replicate == -1].sort_values("x")
            boots = d[d.replicate >= 0].pivot_table(index="replicate", columns="x", values="ALE")
            lo, hi = np.nanpercentile(boots.to_numpy(), [2.5, 97.5], axis=0) if len(boots) else (None, None)
            ax, axn = P.effect_panel(fig, g[j], base.x, base.ALE, color=TCOL[t], xlabel=LABELS[k])
            if lo is not None:
                xs = boots.columns.to_numpy(float)
                ax.fill_between(xs, lo, hi, color=TCOL[t], alpha=0.18, lw=0)
            axn.hist(T("train")[k], bins=20, color=F.tint(F.INK, 0.75))
            if j == 0:
                ax.set_ylabel(f"ALE, {TSHORT[t]}")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_ylim(*ylims[k])
            F.panel(fig, ax, "abcdef"[3 * row + j], f"{SHORT[k]}, {'all vehicles' if row == 0 else 'general'}")
    F.save(fig, out / "Fig08_ALE_response_shapes")


# =========================================================================================
def fig08_mechanism(T, out):
    cell = T("mechanism_cells")
    dec = T("mechanism_variance_shares")
    con = T("mechanism_headway_contrasts")
    fig = F.new_figure(180, 165)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1])
    g0 = gs[0].subgridspec(1, 2)
    bg = "midpoint"
    for j, t in enumerate(DELAYS):
        ax = fig.add_subplot(g0[j])
        c = cell[cell.background == bg].pivot_table(index="dv_level", columns="headway_level",
                                                    values=f"{t}_mean")
        dv = cell[cell.background == bg].groupby("dv_level").departureVariation.first()
        bh = cell[cell.background == bg].groupby("headway_level").busHeadway.first()
        im = ax.imshow(c.to_numpy(), origin="lower", cmap=F.SEQ, aspect="auto")
        ax.set_xticks(range(len(bh)), [f"{v / 60:.0f}" for v in bh])
        ax.set_yticks(range(len(dv)), [f"{v / 60:.0f}" for v in dv])
        ax.set_xlabel("Bus headway [min]")
        ax.set_ylabel("Departure variation [min]")
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(TSHORT[t], fontsize=8)
        cb.outline.set_visible(False)
        F.panel(fig, ax, "ab"[j], "All-vehicle delay" if j == 0 else "General-traffic delay")
    g1 = gs[1].subgridspec(1, 2, width_ratios=[1.2, 1])
    ax = fig.add_subplot(g1[0])
    m = cell.groupby(["background", "headway_level"]).agg(
        bh=("busHeadway", "first"), all=("mean_time_loss_sec_mean", "mean"),
        gen=("general_time_loss_sec_mean", "mean"), bus=("demand_bus_in_window_mean", "mean")).reset_index()
    for b, ls in [("midpoint", "-"), ("car-oriented", "--")]:
        g = m[m.background == b]
        if g.empty:
            continue
        ax.plot(g.bh / 60, g["all"], ls, marker="o", ms=3.5, color=TCOL["mean_time_loss_sec"],
                label=f"All vehicles, {b}")
        ax.plot(g.bh / 60, g["gen"], ls, marker="s", ms=3.5, color=TCOL["general_time_loss_sec"],
                label=f"General traffic, {b}")
    g = m[m.background == "midpoint"]
    nb0, nb1 = g.bus.iloc[0], g.bus.iloc[-1]
    ax.annotate(f"{nb0:.0f} buses → {nb1:.0f} buses\nin the 4-h period", xy=(g.bh.iloc[-1] / 60, g["all"].iloc[-1]),
                xytext=(0.55, 0.42), textcoords="axes fraction", fontsize=7.4, ha="center",
                arrowprops=dict(arrowstyle="->", color=F.INK2, lw=0.8))
    ax.set_xlabel("Bus headway [min]")
    ax.set_ylabel("Mean time loss [s]")
    ax.legend(fontsize=7, loc="upper right")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Composition effect")
    ax = fig.add_subplot(g1[1])
    comps = ["departureVariation", "busHeadway", "interaction"]
    ccol = [F.S4[0], F.S4[1], F.tint(F.INK, 0.6)]
    bars = [(b, o) for b in ["midpoint", "car-oriented"] for o in DELAYS
            if not dec[(dec.background == b) & (dec.outcome == o)].empty]
    left = np.zeros(len(bars))
    for c, col in zip(comps, ccol):
        v = np.array([dec[(dec.background == b) & (dec.outcome == o) & (dec.component == c)].share.iloc[0]
                      for b, o in bars])
        ax.barh(range(len(bars)), v, left=left, color=col, height=0.6,
                label={"departureVariation": "Departure variation", "busHeadway": "Bus headway",
                       "interaction": "Interaction"}[c])
        for i, (vv, ll) in enumerate(zip(v, left)):
            if vv > 0.08:
                ax.text(ll + vv / 2, i, f"{vv:.0%}", ha="center", va="center", fontsize=7.2,
                        color=F.text_on(col))
        left += v
    ax.set_yticks(range(len(bars)), [f"{'All' if o == DELAYS[0] else 'General'}, {b}" for b, o in bars],
                  fontsize=7.6)
    ax.set_xlim(0, 1)
    F.pct_axis(ax, "x")
    ax.set_xlabel("Share of factorial variance")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    F.clean_left(ax)
    F.panel(fig, ax, "d", "Variance shares")
    F.save(fig, out / "Fig09_headway_mechanism")


# =========================================================================================
def fig09_recourse(T, out):
    v = T("recourse_validation")
    rec = v[v.arm == "recourse"].sort_values("achieved_reduction_pct").reset_index(drop=True)
    rnd = v[v.arm == "random"].set_index("setting_id").reindex(rec.setting_id).reset_index()
    fig = F.new_figure(180, 160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    g0 = gs[0].subgridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(g0[0])
    y = np.arange(len(rec))
    xmax = max(rec.CI_high.max(), rec.predicted_reduction_pct.max()) + 4
    ax.axvspan(15, xmax, color=F.tint(F.G1, 0.92), lw=0)
    ax.set_xlim(min(0, rec.CI_low.min() - 2), xmax)
    ax.text(15.4, len(rec) - 0.6, "15% target", fontsize=7.4, color=F.G1, va="top")
    ax.hlines(y, rec.CI_low, rec.CI_high, color=F.G0, lw=1.4)
    ax.scatter(rec.achieved_reduction_pct, y, s=24, color=F.G0, zorder=3, label="Achieved in SUMO (95% CI)")
    ax.scatter(rec.predicted_reduction_pct, y, s=28, marker="|", color=F.G1, lw=1.6, zorder=4,
               label="Predicted by surrogate")
    ax.axvline(0, color=F.INK2, lw=0.7)
    ax.axvline(15, color=F.G1, lw=1, ls="--")
    ax.set_yticks(y, [s.replace("policy_", "P") for s in rec.setting_id], fontsize=7)
    ax.set_xlabel("All-vehicle delay reduction [%]")
    ax.legend(fontsize=7, loc="lower right")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Predicted vs achieved")
    ax = fig.add_subplot(g0[1])
    lo, hi = _lims(rec.achieved_reduction_pct, rnd.achieved_reduction_pct, [0])
    _diag(ax, lo, hi)
    ax.axhline(15, color=F.G1, lw=0.8, ls=":")
    ax.axvline(15, color=F.G1, lw=0.8, ls=":")
    ax.scatter(rnd.achieved_reduction_pct, rec.achieved_reduction_pct, s=28, color=F.G0, ec=F.BG, zorder=3)
    better = int((rec.achieved_reduction_pct.to_numpy() > rnd.achieved_reduction_pct.to_numpy()).sum())
    ax.text(0.97, 0.05, f"Recourse > random in {better}/{len(rec)} scenarios\n"
            f"target met: {int(rec.delay_target_met.sum())}/{len(rec)} vs {int(rnd.delay_target_met.sum())}/{len(rnd)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4)
    ax.set_xlabel("Random same-distance change [%]")
    ax.set_ylabel("Surrogate recourse [%]")
    F.ygrid(ax)
    F.panel(fig, ax, "b", "Recourse vs random control")
    g1 = gs[1].subgridspec(1, 2)
    ax = fig.add_subplot(g1[0])
    for arm, c in [("recourse", F.G0), ("random", F.MUTED)]:
        d = v[v.arm == arm]
        ax.errorbar(d.achieved_reduction_pct, d.general_reduction_pct,
                    yerr=[d.general_reduction_pct - d.general_CI_low, d.general_CI_high - d.general_reduction_pct],
                    fmt="o", ms=4, color=c, lw=0.8, label=arm.capitalize())
    lo, hi = _lims(v.achieved_reduction_pct, v.general_reduction_pct, [0])
    _diag(ax, lo, hi)
    ax.set_xlabel("All-vehicle delay reduction [%]")
    ax.set_ylabel("General-traffic reduction [%]")
    ax.legend(fontsize=7.2, loc="upper left")
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Who benefits")
    ax = fig.add_subplot(g1[1])
    met = [("completion_ratio", "Completed trips"), ("bus_vehicles_ratio", "Bus vehicles"),
           ("private_trip_ratio", "Private car trips"), ("vkt_ratio", "VKT")]
    data = [rec[c].to_numpy() for c, _ in met]
    P.raincloud(ax, data[::-1], [l for _, l in met][::-1], colors=[F.S4[3], F.S4[2], F.S4[1], F.S4[0]],
                show_stats=False)
    ax.axvline(1, color=F.INK2, lw=0.8)
    ax.axvline(0.95, color=F.G1, lw=0.8, ls="--")
    ax.set_xlabel("Recourse / baseline ratio")
    F.panel(fig, ax, "d", "Side effects")
    F.save(fig, out / "Fig10_recourse_validation")


# =========================================================================================
def fig10_inverse(T, out):
    rec = T("inverse_parameter_recovery")
    eq = T("inverse_equifinality_ranges")
    val = T("inverse_validation")
    objs = ["Aggregate (2 outputs)", "Extended (8 outputs)"]
    ocol = dict(zip(objs, [F.G0, F.G1]))
    fig = F.new_figure(180, 160)
    gs = fig.add_gridspec(2, 1)
    g0 = gs[0].subgridspec(1, 2)
    ax = fig.add_subplot(g0[0])
    order = rec.groupby("parameter").abs_error.median().sort_values().index.tolist()
    y = np.arange(len(order))
    for k, o in enumerate(objs):
        d = rec[rec.objective == o]
        for i, p in enumerate(order):
            v = d[d.parameter == p].abs_error.to_numpy()
            yy = i + (0.17 if k == 0 else -0.17)
            ax.scatter(v, np.full(len(v), yy), s=8, color=F.tint(ocol[o], 0.45), lw=0)
            ax.scatter(np.median(v), yy, s=34, marker="D", color=ocol[o], ec=F.BG, zorder=3,
                       label=o if i == 0 else None)
    ax.axvline(1 / 3, color=F.MUTED, ls=":", lw=1)
    ax.text(1 / 3, len(order) - 0.4, " random guess", fontsize=7, color=F.INK2)
    ax.set_yticks(y, [SHORT[p] for p in order])
    ax.set_xlabel("|Recovered − true| / parameter range")
    ax.legend(fontsize=7, loc="lower right")
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "a", "Parameter recovery")
    ax = fig.add_subplot(g0[1])
    obj_raw = {"aggregate": objs[0], "extended": objs[1]}
    for k, (raw, o) in enumerate(obj_raw.items()):
        d = eq[eq.objective == raw].groupby("parameter").normalized_IQR.median().reindex(order)
        ax.barh(y + (0.2 if k == 0 else -0.2), d, 0.38, color=ocol[o], label=o)
    ax.set_yticks(y, [SHORT[p] for p in order])
    ax.set_xlabel("IQR of near-optimal set / range")
    ax.axvline(0.5, color=F.MUTED, ls=":", lw=1)
    ax.text(0.5, len(order) - 0.45, "uniform ", ha="right", fontsize=7, color=F.INK2)
    F.clean_left(ax)
    F.ygrid(ax, "x")
    F.panel(fig, ax, "b", "Equifinality")
    ax = fig.add_subplot(gs[1])
    mets = ["mean_time_loss_sec", "total_vkt_km", "general_time_loss_sec", "total_completed_trips",
            "p90_time_loss_sec", "type_default_completed", "mean_waiting_sec"] + \
           [f"arrivals_hour_{h}" for h in [6, 7, 8, 9]]
    lab = {"mean_time_loss_sec": "All\ndelay", "total_vkt_km": "VKT", "general_time_loss_sec": "General\ndelay",
           "total_completed_trips": "Completed\ntrips", "p90_time_loss_sec": "P90\ndelay",
           "type_default_completed": "Car\ntrips", "mean_waiting_sec": "Waiting\ntime",
           "arrivals_hour_6": "Arrivals\n06\u201307", "arrivals_hour_7": "Arrivals\n07\u201308",
           "arrivals_hour_8": "Arrivals\n08\u201309", "arrivals_hour_9": "Arrivals\n09\u201310"}
    for k, o in enumerate(objs):
        d = val[val.objective == o]
        for i, m in enumerate(mets):
            v = d[d.metric == m]
            if v.empty:
                continue
            x = i + (k - 0.5) * 0.36
            inobj = bool(v.in_objective.iloc[0])
            bp = ax.boxplot(np.clip(v.relative_error_pct, 0.1, None), positions=[x], widths=0.3, patch_artist=True,
                            showfliers=False, medianprops=dict(color=F.INK))
            bp["boxes"][0].set(facecolor=ocol[o] if inobj else F.BG, edgecolor=ocol[o], lw=1.1)
    ax.set_xticks(range(len(mets)), [lab[m] for m in mets])
    ax.set_ylabel("Fresh-seed relative error [%]")
    F.plain_log(ax, [0.1, 1, 10, 100], axis="y")
    ax.set_ylim(0.08, 400)
    ax.legend(handles=[Patch(fc=ocol[o], label=f"{o}, in objective") for o in objs] +
              [Patch(fc=F.BG, ec=F.INK2, label="Not in objective")], fontsize=7.3, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.2))
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Calibrated settings re-simulated")
    F.save(fig, out / "Fig11_inverse_calibration")


# =========================================================================================
def fig11_transfer_robustness(T, out):
    tp = T("transfer_frozen_predictions")
    lr = T("adaptation_learning")
    tel = T("teleport_pairs")
    mfd = T("routing_mfd")
    fig = F.new_figure(180, 155)
    gs = fig.add_gridspec(2, 1)
    g0 = gs[0].subgridspec(1, 2)
    ax = fig.add_subplot(g0[0])
    t = "general_time_loss_sec"
    e = tp[(tp.target == t) & (tp.model == "Ensemble")]
    lo, hi = _lims(e.observed, e.predicted)
    _diag(ax, lo, hi)
    ax.scatter(e.observed, e.predicted, s=12, color=F.G0, lw=0)
    from .surrogates import metrics
    mm = metrics(e.observed, e.predicted)
    ax.text(0.04, 0.96, f"R² {mm['R2']:.2f}\nbias {mm['bias']:+.1f} s", transform=ax.transAxes,
            va="top", fontsize=7.6)
    ax.set_xlabel("8×8 SUMO, general delay [s]")
    ax.set_ylabel("Frozen 6×6 ensemble [s]")
    F.ygrid(ax)
    F.panel(fig, ax, "a", "Frozen transfer")
    ax = fig.add_subplot(g0[1])
    mcol = {"Frozen_source": F.MUTED, "Affine_adaptation": F.S4[2], "Residual_adaptation": F.S4[0],
            "Target_only": F.S4[1]}
    d = lr[lr.target == t]
    for m, c in mcol.items():
        g = d[d.method == m]
        ax.plot(g.n_adaptation, g.RMSE, "-o", ms=3.5, color=c, label=m.replace("_", " "))
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(d.n_adaptation.unique()), [str(v) for v in sorted(d.n_adaptation.unique())])
    ax.set_xlabel("8×8 adaptation settings")
    ax.set_ylabel("CV RMSE, general delay [s]")
    ax.legend(fontsize=7, loc="center right", bbox_to_anchor=(1.0, 0.62))
    F.ygrid(ax)
    F.panel(fig, ax, "b", "Adaptation budget")
    g1 = gs[1].subgridspec(1, 2)
    ax = fig.add_subplot(g1[0])
    am, ai = T("adaptation_test_metrics"), T("adaptation_intervals")
    meths = list(mcol)
    x = np.arange(len(meths))
    for k, t in enumerate(DELAYS):
        r = am[am.target == t].set_index("method").reindex(meths)
        c = ai[ai.target == t].set_index("method").reindex(meths)
        xx = x + (k - 0.5) * 0.38
        ax.bar(xx, r.RMSE, 0.36, color=[F.tint(mcol[m], 0.0 if k == 0 else 0.45) for m in meths],
               edgecolor=[mcol[m] for m in meths], lw=0.8)
        for xi, (rm, cv) in zip(xx, zip(r.RMSE, c.coverage)):
            if rm > 15:
                ax.text(xi, 15.5, f"{rm:.0f} s\n{cv:.0%}", ha="center", va="top", fontsize=6.8,
                        color="white" if k == 0 else F.INK, fontweight="bold")
            else:
                ax.text(xi, rm + 0.3, f"{cv:.0%}", ha="center", va="bottom", fontsize=6.8, rotation=90)
    ax.set_ylim(0, 16)
    ax.set_xticks(x, [m.replace("_adaptation", "").replace("_", " ") for m in meths])
    ax.set_ylabel("Fresh 8\u00d78 test RMSE [s]")
    ax.legend(handles=[Patch(fc=F.MUTED, label="All-vehicle delay"), Patch(fc=F.tint(F.MUTED, 0.45), ec=F.MUTED, label="General delay")],
              fontsize=7, loc="upper right", title="Label: 90% interval coverage", title_fontsize=6.8)
    F.ygrid(ax)
    F.panel(fig, ax, "c", "Adapted models, 8\u00d78")
    ax = fig.add_subplot(g1[1])
    for mode, c in [("static", F.G0), ("adaptive", F.G1)]:
        g = mfd[mfd["mode"] == mode].sort_values("time_h")
        pts = ax.scatter(g.density, g.flow, s=14, color=c, label=mode.capitalize(), zorder=3)
        ax.plot(g.density, g.flow, color=F.tint(c, 0.5), lw=0.9)
    ax.set_xlabel("Employment-zone density [veh/km]")
    ax.set_ylabel("Flow [veh/h]")
    ax.legend(fontsize=7.2, title="Routing", title_fontsize=7.2)
    F.ygrid(ax)
    F.panel(fig, ax, "d", "Network state")
    F.save(fig, out / "Fig12_transfer_and_robustness")


FIGURES = [fig00_framework, fig01_design, fig02_noise, fig03_development, fig04_test, fig05_intervals, fig06_sobol,
           fig07_ale, fig08_mechanism, fig09_recourse, fig10_inverse, fig11_transfer_robustness]


def make_all(study_or_root, only=None):
    root = Path(getattr(study_or_root, "root", study_or_root))
    out = root / "figures"
    T = Tables(root)
    done = []
    for f in FIGURES:
        if only and not any(o in f.__name__ for o in only):
            continue
        F.use_theme("classic")
        f(T, out)
        done.append(f.__name__)
        print("figure:", f.__name__, flush=True)
    return done


if __name__ == "__main__":
    make_all(sys.argv[1], sys.argv[2:] or None)

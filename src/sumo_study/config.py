"""Study-wide constants: parameter domain, outputs, experiment sizes and seed registry.

Everything that defines *what* is simulated lives here, so the experimental design can be
read in one place and cited in the manuscript. Changing a value here changes the simulator
fingerprint (see ``simulator.py``) and therefore never silently reuses incompatible runs.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------
# Nine activity-based demand and transit-supply inputs (physical units, uniform domain).
# --------------------------------------------------------------------------------------------
BOUNDS: dict[str, tuple[float, float]] = {
    "carRate": (0.20, 0.90),               # share of households owning a car
    "carPreference": (0.10, 0.95),         # propensity to drive when a car is available
    "unemploymentRate": (0.02, 0.25),
    "freeTimeActivityRate": (0.05, 0.50),
    "departureVariation": (60.0, 1800.0),  # s, SD of departure times around activity starts
    "footDistanceLimit": (150.0, 1200.0),  # m, maximum walking distance
    "busHeadway": (300.0, 2400.0),         # s, interval between bus departures
    "uniformRandomTraffic": (0.01, 0.35),  # share of non-activity random traffic
    "morningPeakProp": (0.30, 0.85),       # share of jobs opening at 08:00 (rest 09:00)
}
FEATURES: list[str] = list(BOUNDS)
LOW = np.array([BOUNDS[k][0] for k in FEATURES])
HIGH = np.array([BOUNDS[k][1] for k in FEATURES])

LABELS: dict[str, str] = {
    "carRate": "Car ownership rate",
    "carPreference": "Car preference",
    "unemploymentRate": "Unemployment rate",
    "freeTimeActivityRate": "Free-time activity rate",
    "departureVariation": "Departure variation [s]",
    "footDistanceLimit": "Walking limit [m]",
    "busHeadway": "Bus headway [s]",
    "uniformRandomTraffic": "Random traffic share",
    "morningPeakProp": "Morning-peak share",
}
SHORT: dict[str, str] = {
    "carRate": "Car rate", "carPreference": "Car pref.", "unemploymentRate": "Unemploy.",
    "freeTimeActivityRate": "Free-time", "departureVariation": "Dep. var.",
    "footDistanceLimit": "Walk limit", "busHeadway": "Headway",
    "uniformRandomTraffic": "Random traf.", "morningPeakProp": "Peak share",
}

# Primary surrogate outputs: all-vehicle time loss, VKT and general-traffic time loss, which
# excludes scheduled buses (their stop dwell time is counted by SUMO as time loss).
TARGETS: list[str] = ["mean_time_loss_sec", "total_vkt_km", "general_time_loss_sec"]
TARGET_LABELS = {"mean_time_loss_sec": "Mean time loss [s]", "total_vkt_km": "VKT [km]",
                 "general_time_loss_sec": "General-traffic time loss [s]"}
TARGET_SHORT = {"mean_time_loss_sec": "All-vehicle delay", "total_vkt_km": "VKT",
                "general_time_loss_sec": "General-traffic delay"}
TARGET_UNITS = {"mean_time_loss_sec": "s", "total_vkt_km": "km", "general_time_loss_sec": "s"}

# Auxiliary outputs: modelled only for guardrails and the extended calibration objective.
HOURS = [6, 7, 8, 9, 10]
AUX_TARGETS: list[str] = ["total_completed_trips"] + [f"arrivals_hour_{h}" for h in HOURS]

# Levers available to the planner in the recourse analysis.
LEVERS = ["departureVariation", "busHeadway", "footDistanceLimit"]

# --------------------------------------------------------------------------------------------
# Fixed simulation physics. Part of the simulator fingerprint.
# --------------------------------------------------------------------------------------------
PHYSICS = dict(
    begin=23400,          # 06:30
    end=37800,            # 10:30
    teleport=300,         # s; SUMO default rule retained, tested in the teleport stage
    grid_number=6,        # 6 x 6 junction grid -> 120 directed edges
    block_length=250,     # m
    lanes=2,
    speed=13.89,          # m/s (50 km/h)
    inhabitants=3500,
    incomingTraffic=450,
)

# --------------------------------------------------------------------------------------------
# Experiment sizes. "smoke" checks the full pipeline in minutes; "full" is the paper.
# --------------------------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    "smoke": dict(
        n_train=32, n_cal=16, n_test=16, test_seeds=2,
        n_rep=4, rep_seeds=3,
        n_recourse=3, policy_seeds=3, search_budget=256,
        n_targets=2, target_seeds=3, calib_trials=60,
        mech_levels=3, mech_seeds=2, mech_backgrounds=2,
        sobol_sim_power=4, saltelli_power=7, n_boot=4,
        n_transfer=6, n_adapt=16, n_adapt_cal=8, n_adapt_test=8,
        n_teleport=4, model_size=60,
    ),
    "full": dict(
        n_train=512, n_cal=200, n_test=200, test_seeds=5,
        n_rep=24, rep_seeds=10,
        n_recourse=20, policy_seeds=10, search_budget=6000,
        n_targets=10, target_seeds=10, calib_trials=600,
        mech_levels=8, mech_seeds=10, mech_backgrounds=2,
        sobol_sim_power=9, saltelli_power=11, n_boot=50,
        n_transfer=100, n_adapt=128, n_adapt_cal=100, n_adapt_test=100,
        n_teleport=24, model_size=500,
    ),
}

# --------------------------------------------------------------------------------------------
# Seed registry. Every experiment draws designs and SUMO seeds from its own disjoint block,
# so no simulation is accidentally shared between partitions (checked in design.py).
# --------------------------------------------------------------------------------------------
DESIGN_SEEDS = dict(train=2026, calibration=47201, test=83003, targets=98013,
                    transfer=77301, adapt=193107, adapt_calibration=193109,
                    adapt_test=193117, sobol_sim=31337)
RUN_SEED_BASE = dict(train=100_000, calibration=200_000, test=300_000, replication=400_000,
                     recourse=500_000, inverse_target=600_000, inverse_validation=700_000,
                     mechanism=800_000, transfer=900_000, sobol_sim=1_000_000,
                     adapt=1_200_000, adapt_calibration=1_300_000, adapt_test=1_400_000)

ALPHA = 0.10            # nominal 90 % prediction intervals
DELAY_TARGET = 0.15     # recourse: >= 15 % reduction in mean time loss
COMPLETION_GUARD = 0.95  # recourse: completed trips must stay >= 95 % of baseline

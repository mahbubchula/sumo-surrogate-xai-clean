"""Generate notebooks/SUMO_Surrogate_Study.ipynb (the ordered, documented study notebook)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md("""
# Explainable SUMO surrogates — validation-first study

This notebook reproduces every result, table and figure of the article, in order:

| § | Analysis | Question |
|---|---|---|
| 3 | Partitioned designs (512 Sobol / 200 / 200×5 seeds) | Independent evaluation data |
| 4 | Development CV and frozen surrogates (incl. Gaussian process) | Model fitting |
| 5 | Independent test + noise ceiling | Accuracy on new settings |
| 6 | Train → calibrate → test predictive intervals | Interval coverage |
| 7 | Repeated-seed replication | Seed noise |
| 8 | Interventions re-simulated with matched seeds + random control | Achieved reductions |
| 9 | Inverse calibration on excluded targets, two objectives | Identifiability |
| 10 | Headway × departure-variation factorial | Transit mechanism |
| 11 | Direct-SUMO Sobol indices (simulator ground truth) | Simulator sensitivity |
| 12 | Refit stability of Sobol/SHAP/ALE; DML projection | Explanation stability |
| 13 | Learning curve vs. noise floor | Sample size |
| 14 | Teleport rule, rerouting, 8×8 transfer and adaptation | Simulator and network robustness |
| 15–16 | Figures and frozen export | Reproducibility |

Every stage is **resumable**: SUMO runs are cached by a hash of (simulator fingerprint,
parameters, seed); completed stages are skipped. The analysis protocol fixed before the full
run is in `docs/ANALYSIS_PROTOCOL.md`.
""")

md("## 1. Environment\nRuns locally (Eclipse SUMO 1.28 on PATH) or on Google Colab (set `ON_COLAB`).")
code("""
import os, sys, subprocess
from pathlib import Path
ON_COLAB = "google.colab" in sys.modules
if ON_COLAB:
    subprocess.run("pip -q install eclipse-sumo==1.28.0 -r requirements.txt", shell=True, check=True)
    import sumo; os.environ["PATH"] = str(Path(sumo.SUMO_HOME) / "bin") + os.pathsep + os.environ["PATH"]
REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO / "src"))
print(subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0])
""")

md("## 2. Study configuration\n`PROFILE = \"smoke\"` checks the whole pipeline in ~3 minutes; `\"full\"` is the paper (≈10,000 SUMO runs; ≈90 min on 8 cores).")
code("""
import pandas as pd, json
from IPython.display import Image, display
from sumo_study.workflow import Study, STAGES
from sumo_study.config import PROFILES, BOUNDS
pd.set_option("display.precision", 3); pd.set_option("display.width", 160)
display(Image(filename=str(REPO / "results" / "full" / "figures" / "png" / "Fig01_methodological_framework.png"), width=900))
PROFILE, WORKERS = "full", 8
study = Study(REPO / "results", profile=PROFILE, workers=WORKERS)
print(json.dumps(PROFILES[PROFILE], indent=1)); pd.DataFrame(BOUNDS, index=["low", "high"]).T
""")
code("""
def ensure(stage):
    \"\"\"Run a stage only if it has not completed (results and SUMO runs are cached).\"\"\"
    if study.status.get(stage, {}).get("status") != "complete":
        study.run(stage)
    else:
        print(f"{stage}: complete ({study.status[stage].get('elapsed_sec', '?')} s)")
T = study.load
def fig(name, width=1400):
    \"\"\"Show a downscaled preview; full-resolution files are in results/<profile>/figures/.\"\"\"
    import io
    from PIL import Image as PILImage
    im = PILImage.open(study.root / "figures" / "png" / f"{name}.png")
    im.thumbnail((width, width))
    buf = io.BytesIO(); im.convert("RGB").save(buf, format="JPEG", quality=85)
    display(Image(data=buf.getvalue(), format="jpeg", width=900))
""")

sections = [
    ("3. Partitioned designs and simulations", "design",
     "Development: scrambled Sobol. Calibration and test: independent uniform samples over the same domain, disjoint, with their own seed blocks; every test setting is run with 5 seeds.",
     "display(T('simulation_audit').groupby('experiment').size().rename('runs').to_frame())\nT('design_nearest_neighbour_distance').groupby('split').describe().T"),
    ("4. Development cross-validation and frozen surrogates", "models",
     "Fixed hyperparameters; models are hashed in `models/freeze_record.json` before calibration or test outcomes are read.",
     "T('dev_cv_summary').pivot(index='model', columns='target', values='R2_mean')"),
    ("5. Independent test of frozen surrogates", "independent_test",
     "Single-run and seed-mean estimands; paired bootstrap error differences.",
     "tm = T('test_metrics'); display(tm[tm.estimand=='single_run'].pivot(index='model', columns='target', values='R2'))\ndisplay(T('noise_test_settings'))\nfig('Fig05_independent_test')"),
    ("6. Predictive intervals: train → calibrate → test", "intervals",
     "CQR, split conformal, native GP and GP-normalised conformal at 90% nominal.",
     "c = T('interval_coverage'); display(c[c.estimand!='seed_mean'][['target','method','estimand','coverage','CI_low','CI_high','mean_width']])\nfig('Fig06_predictive_intervals')"),
    ("7. Repeated-seed replication", "replication",
     "24 development settings spanning the delay range × 10 seeds.",
     "display(T('noise_replication'))\nfig('Fig03_simulation_noise')"),
    ("8. Surrogate-proposed interventions validated in SUMO", "recourse",
     "Surrogate proposals re-simulated with 10 matched seeds against the baseline and a random same-distance control.",
     "display(T('recourse_overview'))\nfig('Fig10_recourse_validation')"),
    ("9. Inverse calibration on excluded targets", "inverse_calibration",
     "Two objectives; fresh-seed re-simulation; dense equifinality scan.",
     "display(T('inverse_parameter_recovery').groupby(['objective','parameter']).abs_error.median().unstack(0))\nfig('Fig11_inverse_calibration')"),
    ("10. Headway × departure-variation factorial", "mechanism",
     "8×8 levels × 10 common seeds at two backgrounds; outcomes separated by vehicle type.",
     "display(T('mechanism_variance_shares').pivot_table(index=['background','outcome'], columns='component', values='share'))\nfig('Fig09_headway_mechanism')"),
    ("11. Direct-SUMO Sobol indices", "sobol_sumo",
     "Radial A/B/AB_i design on the simulator itself; seed-noise-corrected estimators.",
     "s = T('sobol_sumo_indices'); s.pivot_table(index='parameter', columns=['target','estimator'], values='ST').round(3)"),
    ("12. Surrogate sensitivity, refit stability, SHAP, ALE, DML", "sensitivity",
     "50 bootstrap refits of the ensemble; interventional ensemble TreeSHAP; ALE with refit bands.",
     "display(T('sobol_rank_stability').groupby('target')[['kendall_tau','top2_same']].mean())\nfig('Fig07_sobol_sensitivity'); fig('Fig08_ALE_response_shapes')"),
    ("13. Learning curve against the noise floor", "learning_curve", "Development folds only.",
     "fig('Fig04_development_and_learning')"),
    ("14a. Teleport rule", "teleport", "Same settings and seeds with teleporting disabled.", "T('teleport_pairs')"),
    ("14b. Static vs adaptive rerouting", "routing", "Median-delay development setting.", "T('routing_mfd').groupby('mode').mean()"),
    ("14c. Transfer to an 8×8 grid and adaptation", "transfer", "Frozen transfer, then four prespecified adaptation methods.",
     "display(T('adaptation_test_metrics'))\nfig('Fig12_transfer_and_robustness')"),
    ("14d. Runtime and simulation-quality audit", "runtime_quality", "All runs, teleports, completion and surrogate speed-up.",
     "display(T('simulation_audit_by_experiment')); print(json.dumps(json.load(open(study.root/'tables'/'runtime.json')), indent=1))"),
    ("15. Manuscript figures", "figures", "Figure-studio classic theme, 180 mm, PNG 600 dpi + PDF/SVG/TIFF.",
     "fig('Fig02_design_and_network')"),
    ("16. Frozen export", "export", "Key numbers, LaTeX tables, SHA-256 manifest and results archive.",
     "print((study.root/'artifact_manifest.csv').read_text()[:800])"),
]
for title, stage, text, show in sections:
    md(f"## {title}\n{text}")
    code(f"ensure('{stage}')\n{show}")

md("""
## Boundaries that remain
* One synthetic grid family (6×6, with an 8×8 stress test); edge-order-based zones.
* activitygen emits vehicle trips only; bus outcomes are bus *vehicles*, not passengers.
* Simulator-internal effects are not real-world causal policy effects; no observed traffic data.
""")
nb["cells"] = C
nb["metadata"]["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
out = Path(__file__).resolve().parents[1] / "notebooks" / "SUMO_Surrogate_Study.ipynb"
nbf.write(nb, out)
print(out)

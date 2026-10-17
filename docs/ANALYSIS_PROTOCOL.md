# Analysis protocol (fixed before the full experiment)

This protocol was committed before the `full` profile was executed. Its purpose is to make
clear which choices were fixed in advance, so that independent-test and validation results
cannot have influenced model or method selection.

## Estimands
* **Primary prediction target:** the outcome of a single future SUMO run at a new parameter
  setting (one activitygen/duarouter/sumo seed). The seed-mean response is reported as a
  secondary, explicitly different estimand.
* **Outputs:** mean time loss of completed vehicles (`mean_time_loss_sec`),
  completed-trip VKT (`total_vkt_km`), and general-traffic time loss
  excluding scheduled buses (`general_time_loss_sec`, added because SUMO counts bus dwell
  time at stops as time loss, which makes the all-vehicle mean depend on the number of buses).

## Partitions (all from the same uniform input domain, disjoint, own seed blocks)
| Partition | Design | Settings | Seeds per setting | Use |
|---|---|---|---|---|
| Development | scrambled Sobol (2^9) | 512 | 1 | CV, fitting, SHAP/ALE/Sobol on surrogate |
| Calibration | independent uniform | 200 | 1 | interval calibration only |
| Test | independent uniform | 200 | 5 | evaluated once, after models were frozen |

## Fixed modelling choices
* Model zoo with fixed hyperparameters: Ridge (standardised), Gaussian process (Matérn 5/2 ARD
  + white noise), Random forest, LightGBM, CatBoost, XGBoost; **primary surrogate = equal-weight
  CatBoost + XGBoost ensemble**. No tuning on calibration or
  test data. Models are frozen and hashed (`models/freeze_record.json`) before any calibration
  or test outcome is read.
* Interval methods at 90 % nominal: CQR (LightGBM quantiles), split conformal on ensemble
  residuals, native GP intervals, GP-normalised split conformal. Calibration set only.
  Coverage reported with Wilson (single run) or setting-cluster bootstrap (all runs) CIs.

## Validation experiments
1. **Simulation noise:** 24 development settings spanning delay quantiles x 10 seeds; and the
   200 test settings x 5 seeds. Within/between variance, R^2 ceiling and RMSE floor.
2. **Recourse:** the 20 highest-delay development settings; levers = departure variation, bus
   headway, walking limit; search = 6,000 global/local candidates for the minimum normalised
   change with predicted delay reduction >= 15 % and predicted completed trips >= 95 %.
   Arms baseline / recourse / random direction of equal distance, 10 matched seeds each.
   Reported: achieved reduction (all-vehicle and general-traffic) with paired bootstrap CI,
   target attainment, completion and served-fraction ratios, bus-vehicle ratio.
3. **Inverse calibration:** 10 excluded targets x 10 seeds; objectives *aggregate* (2 outputs)
   and *extended* (8 outputs); 600 TPE trials each; 10 fresh validation seeds; parameter
   recovery, equifinality scan (100,000 surrogate evaluations, RMS mismatch <= 5 %).
4. **Mechanism factorial:** departure variation (8 levels) x bus headway (8 levels) x 10 common
   seeds, at a midpoint and a car-oriented background; outcomes by vehicle type.
5. **Direct-SUMO Sobol:** radial A/B/AB_i design with N = 512 (5,632 runs), Saltelli S1 and
   Jansen ST with and without seed-noise correction, compared with surrogate Sobol on the same
   design; surrogate Sobol with 50 bootstrap refits for rank stability.
6. **Robustness:** teleporting disabled (24 settings, same seeds); static vs adaptive rerouting;
   frozen transfer to an 8x8 grid (100 settings) and prespecified adaptation (128/100/100).

## Decision rules declared in advance
* Recourse success: achieved all-vehicle delay reduction >= 15 % and completed trips >= 95 % of
  baseline (joint criterion). Results are reported whatever their direction.
* No simulation is excluded; failures stop the pipeline. Teleports/unfinished vehicles are
  audited and reported.
* All results are reported, whatever their direction.

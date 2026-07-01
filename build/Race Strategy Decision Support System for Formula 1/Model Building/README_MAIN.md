# Formula 1 Race Strategy Decision Support System

## 1. Project Purpose

This project builds a Formula 1 race strategy decision support system using lap-level race data from the 2019 to 2025 seasons. The aim is to support race strategy decisions by predicting pace, tyre degradation, Safety Car risk, pit stop behaviour, compound choice, traffic effects, rejoin consequences, full-race strategy outcomes, and final risk-aware strategy recommendations.

The system is not built as one single black-box model. Instead, it is designed as a multi-model pipeline. Each model solves one important part of the race strategy problem, and the final models combine those outputs to simulate and recommend race strategy decisions.

The final question answered by the system is:

> Given a race, driver, lap, and scenario, what strategy should be recommended, what result can be expected, and how risky is the recommendation?

---
2. High-Level Pipeline

The full pipeline works in this order:


Raw FastF1 race data
    ↓
Data extraction, cleaning, merging, and feature engineering
    ↓
Model 1: Lap time, pace, tyre degradation, uncertainty
    ↓
Model 2: Safety Car, Virtual Safety Car, Red Flag probabilities
    ↓
Model 3: Pit probability and compound choice probability
    ↓
Model 4: Clean air, position delta, rejoin and traffic effects
    ↓
Final combined master file for Model 5
    ↓
Model 5A: Local pit/wait strategy cost estimation
    ↓
Model 5B: Scenario-based race strategy simulator
    ↓
Model 6: Final risk-aware strategy recommendation layer


Models 1 to 4 create predictive race features. These outputs are merged into one final master dataset. Model 5A estimates local strategy costs. Model 5B simulates strategy outcomes. Model 6 selects the final recommendation using a risk-aware decision score.

---

## 3. Main Data File

The most important final data file is:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
```

This file contains the final lap-level dataset with the outputs of Models 1, 2, 3, and 4 already merged. Models 5A, 5B, and 6 use this file instead of separately loading every earlier trained model.

This file includes information such as:

```text
race_id
driver_id
lap_number
position
tyre_compound
tyre_age
laps_remaining
expected_lap_time
expected_lap_time_fresh
tyre_degradation_delta
tyre_degradation_per_lap
p_sc_next3 / p_sc_next5 / p_sc_next10
p_vsc_next3 / p_vsc_next5 / p_vsc_next10
p_rf_next3 / p_rf_next5 / p_rf_next10
p_pit_next1 / p_pit_next3 / p_pit_next5
p_compound_SOFT / MEDIUM / HARD / INTERMEDIATE / WET
clean_air_flag
traffic and rejoin features
position interaction features
```

---

## 4. Main Scripts

The main Python scripts used in the final project are:

```text
train_model1_lap_time_degradation.py
train_model2_run_final9.py
train_model3a_pit_propensity.py
train_model3b_compound_choice.py
apply_model3c_wet_compound_handler.py
train_model4a_clean_air.py
train_model4b_position_delta.py
train_model4c_rejoin_penalty.py
run_model5a_local_strategy.py
run_model5b_scenario_simulator.py
run_model6.py
```

If your local filenames are slightly different, update this README to match the exact script names in your folder.

---

## 5. Model Summary

Model 1 predicts expected lap time, fresh-tyre pace, tyre degradation, and lap-time uncertainty. It gives the system the basic pace and degradation layer.

Model 2 predicts the probability of future Safety Car, Virtual Safety Car, and Red Flag events over multiple horizons. It gives the system a race interruption risk layer.

Model 3 predicts pit stop probability and tyre compound choice probability. It gives the system pit timing and compound realism.

Model 4 predicts clean air, position movement, traffic pressure, and rejoin penalty. It gives the system track-position and traffic realism.

Model 5A estimates local pit/wait strategy costs using the outputs from Models 1 to 4. It gives Model 5B realistic local strategy cost inputs.

Model 5B simulates full race strategy scenarios using Monte Carlo-style simulation. It compares strategies such as stay out, pit now, wait one lap, wait three laps, wait five laps, and switch to different compounds.

Model 6 is the final recommendation layer. It runs or reads Model 5B, ranks strategies using risk-aware utility, and produces technical and layman-readable reports.

---

## 6. Final Model 6 Run Command

The final pipeline should usually be run through Model 6:

```bash
python run_model6.py \
  --model5b_script run_model5b_scenario_simulator.py \
  --input_path outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv \
  --model5a_strategy_table outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv \
  --outputs_dir outputs \
  --race_id 2019_02_R \
  --driver_id norris \
  --decision_lap 28 \
  --scenario base \
  --strategies auto \
  --n_sim 5000 \
  --seed 42 \
  --risk_appetite balanced
```

When this command is run, Model 6 automatically runs Model 5B first, stores the Model 5B output inside the Model 6 run folder, and then produces the Model 6 final recommendation.

---

## 7. Final Output Folder Structure

A final Model 6 run creates this structure:

```text
outputs/model6_runs/<race_driver_lap_scenario_risk_timestamp>/
    model5b_run/
        results.csv
        metrics.json
        config.json
        report.md

    model6_result/
        model6_ranked_strategies.csv
        model6_recommendation.json
        model6_technical_report.md
        model6_layman_report.md
        model6_combined_report.md
```

The `model5b_run` folder contains the raw scenario simulation. The `model6_result` folder contains the final risk-aware recommendation.

---

## 8. Scenario Options

The strategy simulator supports scenarios such as:

```text
base
no_neutralisation
force_sc_next3
force_sc_next5
force_vsc_next3
force_rf_next3
wet_track
rain_now
rain_from_lap35
rain_from_lap40
```

The `base` scenario uses normal predicted probabilities. The forced scenarios are used for what-if strategy testing.

---

## 9. Strategy Options

When `--strategies auto` is used, the system tests common strategy candidates such as:

```text
stay_out
pit_now_to_SOFT
pit_now_to_MEDIUM
pit_now_to_HARD
pit_now_to_INTERMEDIATE
pit_now_to_WET
wait_1_then_pit_to_SOFT
wait_1_then_pit_to_MEDIUM
wait_1_then_pit_to_HARD
wait_3_then_pit_to_SOFT
wait_3_then_pit_to_MEDIUM
wait_3_then_pit_to_HARD
wait_5_then_pit_to_SOFT
wait_5_then_pit_to_MEDIUM
wait_5_then_pit_to_HARD
```

Wet and intermediate strategies are also included, but they are penalised in dry scenarios unless the scenario justifies them.

---

## 10. Evaluation Approach

The project evaluates each model separately and also evaluates the final pipeline as a full strategy system.

Regression models are evaluated using MAE, RMSE, and R². Classification and probability models are evaluated using ROC-AUC, PR-AUC, Brier score, and log loss. Compound choice is evaluated using log loss, top-1 accuracy, and top-2 accuracy. Model 5B and Model 6 are evaluated using strategy realism, expected finish, expected points, top 10 probability, top 5 probability, podium probability, win probability, downside risk, and comparison against real race scenarios.

---

## 11. Data Leakage Prevention

The project avoids data leakage by using race-level splitting instead of random lap-level splitting. This prevents laps from the same race appearing in both training and testing data.

The target variables are created using future labels, but only current or past features are used as inputs. Future race outcomes, future pit stops, future Safety Cars, and final race results are not used as input features when making predictions at a decision lap.

This is important because the system is designed to behave like a real race strategy tool, where only information available at the current lap should be used.

---

## 12. Research Value

This project is an academic machine learning research project because it defines a clear problem, builds a structured methodology, uses historical data, compares against baselines, avoids data leakage, evaluates models with proper metrics, and validates the final system using realistic race scenarios.

The key contribution is that Formula 1 strategy is treated as a modular decision support problem rather than a single prediction task. Each model contributes one meaningful part of the final race strategy recommendation.

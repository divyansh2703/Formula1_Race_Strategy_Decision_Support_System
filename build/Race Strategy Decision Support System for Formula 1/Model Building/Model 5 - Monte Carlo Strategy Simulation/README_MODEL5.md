# Model 5 README: Strategy Cost and Scenario Simulation

## 1. Purpose

Model 5 turns the earlier predictive model outputs into race strategy outcomes. It is divided into Model 5A and Model 5B.

Model 5 answers this question:

> If the driver chooses a strategy from this lap onward, what race outcome could happen?

---

## 2. Main Scripts

Recommended script names:

```text
run_model5a_local_strategy.py
run_model5b_scenario_simulator.py
```

---

## 3. Model 5A: Local Strategy Cost

Model 5A estimates local pit and wait strategy costs. It compares actions such as:

```text
pit_now
wait_1
wait_3
wait_5
```

It uses information from Models 1 to 4, including pace, degradation, Safety Car probability, pit probability, compound probability, traffic, rejoin effects, and pit lane loss.

The main output file is usually:

```text
outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv
```

This file is used by Model 5B to ground pit cost and waiting cost in the local race situation.

---

## 4. Model 5B: Scenario-Based Simulator

Model 5B is the full strategy simulator. It uses the final master dataset and the Model 5A strategy table to simulate different strategies.

It can test strategies such as:

```text
stay_out
pit_now_to_SOFT
pit_now_to_MEDIUM
pit_now_to_HARD
pit_now_to_INTERMEDIATE
pit_now_to_WET
wait_1_then_pit_to_SOFT
wait_3_then_pit_to_MEDIUM
wait_5_then_pit_to_HARD
```

---

## 5. Inputs

Model 5B uses:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv
```

The first file contains all Model 1 to Model 4 outputs. The second file contains Model 5A local strategy costs.

---

## 6. Scenarios

Model 5B supports scenario testing such as:

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

This allows the system to test how strategy changes under Safety Car, VSC, no-neutralisation, and rain conditions.

---

## 7. Outputs

Model 5B creates:

```text
results.csv
metrics.json
config.json
report.md
```

Important output values include:

```text
mean_projected_race_time
median_projected_race_time
p90_projected_race_time
cvar90_projected_race_time
mean_finish_position
p90_finish_position
mean_points
p_points
p_win
p_podium
p_top5
p_top10
pit_cost_used
pit_cost_source
compound_probability
target_mean_warmup_crossover_cost
target_mean_model3_confidence_cost
```

---

## 8. How Model 5B Works

Model 5B uses Monte Carlo-style simulation. It does not produce only one fixed result. Instead, it simulates many possible race outcomes using pace uncertainty, degradation, event probabilities, pit cost, compound choice probability, traffic, and weather logic.

This allows the model to estimate both average outcome and downside risk.

---

## 9. Connection to Model 6

Model 5B output is the direct input to Model 6.

When Model 6 is run in final pipeline mode, it automatically runs Model 5B and stores the Model 5B output inside:

```text
outputs/model6_runs/<run_name>/model5b_run/
```

Model 6 then reads the Model 5B `metrics.json` file and creates the final risk-aware recommendation.

---

## 10. Why Model 5 Matters

Model 5 is the bridge between prediction and decision-making. Models 1 to 4 predict race components. Model 5 combines those components into actual strategy outcomes.

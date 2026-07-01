# Model 6 README: Final Risk-Aware Strategy Recommendation Layer

## 1. Purpose

Model 6 is the final decision layer of the Formula 1 race strategy pipeline. It runs or reads Model 5B simulation results and selects the final strategy recommendation.

Model 6 answers this question:

> After simulating all strategy options, which strategy should be recommended and how risky is it?

---

## 2. Main Script

Recommended script name:

```text
run_model6.py
```

---

## 3. What Model 6 Does

Model 6 can run in two ways.

In full pipeline mode, it automatically runs Model 5B first. Then it reads the Model 5B result and creates the final Model 6 recommendation.

In existing-result mode, it reads an already-created Model 5B `metrics.json` file and creates a Model 6 recommendation from it.

---

## 4. Main Input Files

Model 6 uses:

```text
run_model5b_scenario_simulator.py
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
outputs/model5a_local_strategy_FINAL_strategy_table_20260609_101034.csv
```

If running from an existing Model 5B result, it uses:

```text
outputs/model5b_runs/<run_name>/metrics.json
```

---

## 5. Main Run Command

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

---

## 6. Risk Appetite Options

Model 6 supports three risk modes:

```text
conservative
balanced
aggressive
```

`conservative` gives more importance to avoiding downside risk.

`balanced` gives the best general trade-off between result and risk.

`aggressive` gives more importance to upside such as top 5, podium, win probability, and expected points.

---

## 7. How Model 6 Calculates Risk

Model 6 calculates risk using:

```text
bad_result_risk
downside_position_spread
tail_time_risk
compound_confidence_score
```

For a front-running driver, bad-result risk is based on missing the podium.

For a points-fighting driver, bad-result risk is based on missing the top 10.

For a recovery driver, bad-result risk is based on missing the points.

---

## 8. How Model 6 Calculates Confidence

Model 6 confidence is based on the gap between the top strategy and the second-best strategy.

It checks:

```text
utility_gap = best utility score - second best utility score
points_gap = best expected points - second best expected points
```

If the gap is large, confidence is high. If the gap is moderate, confidence is medium. If the gap is small, confidence is low.

---

## 9. Outputs

Model 6 creates this folder structure:

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

---

## 10. Important Output Files

`model6_ranked_strategies.csv` contains all strategies ranked by Model 6 utility.

`model6_recommendation.json` contains the structured final recommendation.

`model6_technical_report.md` explains the result in technical language.

`model6_layman_report.md` explains the result in simple language.

`model6_combined_report.md` combines the technical and simple explanation.

---

## 11. Connection to Other Models

Model 6 is connected to all previous models through Model 5B.

Models 1 to 4 create the predictive race features. Model 5A creates local strategy costs. Model 5B simulates all strategies. Model 6 reads those simulation outcomes and chooses the final risk-aware recommendation.

---

## 12. Why Model 6 Matters

Model 5B may identify the raw best simulated strategy, but the raw best strategy is not always the safest or most realistic final choice. Model 6 adds risk-awareness, confidence, and strategy interpretation. This makes it the final decision support layer of the project.

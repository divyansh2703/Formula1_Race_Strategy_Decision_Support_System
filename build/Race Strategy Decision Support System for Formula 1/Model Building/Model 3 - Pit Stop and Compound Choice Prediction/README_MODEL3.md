# Model 3 README: Pit Propensity and Compound Choice

## 1. Purpose

Model 3 predicts pit stop behaviour and tyre compound choice. It helps the system understand when a driver is likely to pit and which compound is realistic.

Model 3 answers this question:

> Is a pit stop likely soon, and which tyre compound makes sense for the next stint?

---

## 2. Main Scripts

Recommended script names:

```text
train_model3a_pit_propensity.py
train_model3b_compound_choice.py
apply_model3c_wet_compound_handler.py
```

---

## 3. Sub-Models

Model 3 is divided into three parts:

```text
Model 3A: Pit propensity model
Model 3B: Dry compound choice model
Model 3C: Wet/intermediate compound handler
```

---

## 4. Model 3A: Pit Propensity

Model 3A predicts the probability that a driver will pit soon. It produces:

```text
p_pit_next1
p_pit_next3
p_pit_next5
```

These mean the probability of a pit stop in the next 1, 3, or 5 laps.

This is useful because pit timing is one of the most important race strategy decisions. Model 5B uses this information to understand whether pit-now, wait-one, wait-three, or wait-five strategies are realistic.

---

## 5. Model 3B: Dry Compound Choice

Model 3B predicts dry tyre compound probability:

```text
p_compound_SOFT
p_compound_MEDIUM
p_compound_HARD
```

This tells the system which dry tyre is realistic. A soft tyre may be realistic late in the race. A hard tyre may be realistic for a long stint. A medium tyre may be a balanced option.

These probabilities are used later as compound confidence scores.

---

## 6. Model 3C: Wet Compound Handler

Model 3C handles wet-weather compounds:

```text
p_compound_INTERMEDIATE
p_compound_WET
```

Intermediate and wet tyres should not be treated like normal dry compounds. In dry conditions, wet tyres should receive a strong penalty. In wet conditions, dry tyres should become risky or unrealistic.

Model 3C helps Model 5B avoid unrealistic tyre choices in rain and dry scenarios.

---

## 7. Inputs

Important features include:

```text
race_id
driver_id
lap_number
laps_remaining
position
tyre_compound
tyre_age
stint_number
pit_stop_number
expected_lap_time
tyre_degradation_delta
safety car probabilities
weather and rain indicators
track temperature
gaps and traffic features
```

---

## 8. Machine Learning Method

Model 3 uses classification models. XGBoost is suitable because pit and compound decisions depend on non-linear relationships between tyre age, laps remaining, compound, race phase, track position, degradation, and weather.

Historical pit rate and majority compound rules are used as baselines.

---

## 9. Evaluation Metrics

Pit propensity is evaluated using:

```text
ROC-AUC
PR-AUC
Brier score
Log loss
```

Compound choice is evaluated using:

```text
Log loss
Top-1 accuracy
Top-2 accuracy
```

Top-2 accuracy is important because more than one compound can be strategically reasonable.

---

## 10. Connection to Other Models

Model 3 outputs are merged into:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
```

Model 5B uses these probabilities to simulate realistic pit and compound choices.

Model 6 uses compound probability as part of the final risk-aware confidence score.

---

## 11. Why Model 3 Matters

Without Model 3, the simulator may choose unrealistic pit timings or tyre compounds. Model 3 gives the pipeline a tyre strategy realism layer. This is especially important when comparing dry, wet, and mixed-weather strategies.

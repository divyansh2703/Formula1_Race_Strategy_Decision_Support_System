# Model 4 README: Clean Air, Position Delta, Rejoin and Traffic Effects

## 1. Purpose

Model 4 models on-track interaction. It helps the system understand whether a driver has clean air, whether position is likely to change, and what penalty may occur after a pit stop rejoin.

Model 4 answers this question:

> If the driver stays out or pits, how will traffic, clean air, and track position affect the strategy?

---

## 2. Main Scripts

Recommended script names:

```text
train_model4a_clean_air.py
train_model4b_position_delta.py
train_model4c_rejoin_penalty.py
```

---

## 3. Sub-Models

Model 4 is divided into three parts:

```text
Model 4A: Clean air prediction
Model 4B: Position delta prediction
Model 4C: Rejoin penalty / traffic loss prediction
```

---

## 4. Model 4A: Clean Air Prediction

Model 4A predicts whether the driver is likely to have clean air.

Clean air means the driver is not stuck behind another car and can use the true pace of the car. Dirty air and traffic can slow the driver down, increase tyre overheating, and make overtaking harder.

Model 4A helps the simulator understand whether a strategy will release the driver into free air or traffic.

---

## 5. Model 4B: Position Delta Prediction

Model 4B predicts short-term position change. It estimates whether the driver is likely to gain, lose, or hold position over the next lap or short horizon.

This helps the strategy system understand local race momentum. A driver who is likely to lose position may need a more aggressive strategy. A driver who is stable may prefer a safer option.

---

## 6. Model 4C: Rejoin Penalty Prediction

Model 4C estimates the time or position penalty caused by rejoining the race after a pit stop.

This is important because a pit stop does not only cost pit lane time. The driver may also rejoin behind traffic. If the driver gets stuck behind slower cars, the strategy can lose additional time even with fresh tyres.

Model 4C helps Model 5B avoid overvaluing pit strategies that would rejoin into heavy traffic.

---

## 7. Inputs

Important features include:

```text
position
field_size
gap_ahead
gap_behind
cars_ahead_count
cars_behind_count
local_traffic_count
local_pack_density
clean_air_flag
overtaking_difficulty_index
pit_lane_time_loss
track_length
current tyre compound
tyre age
expected lap time
fresh tyre gain
safety car probabilities
pit probabilities
compound probabilities
weather features
```

---

## 8. Machine Learning Method

Model 4 uses classification and regression models.

Model 4A is a classification problem because clean air is a probability outcome.

Model 4B and Model 4C are regression problems because position change and rejoin penalty are numerical outcomes.

XGBoost is used because traffic and rejoin behaviour are highly non-linear. A small gap may be very important at one circuit but less important at another circuit. Fresh tyre advantage may help overtaking in some conditions but not in others.

---

## 9. Evaluation Metrics

Model 4A is evaluated using:

```text
ROC-AUC
PR-AUC
Brier score
Log loss
```

Model 4B and Model 4C are evaluated using:

```text
MAE
RMSE
R²
```

---

## 10. Connection to Other Models

Model 4 outputs are merged into:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
```

Model 5B uses Model 4 outputs to simulate traffic, overtaking, clean air, and rejoin effects.

Model 6 indirectly uses Model 4 through the final Model 5B strategy outcomes.

---

## 11. Why Model 4 Matters

Without Model 4, the system would only compare tyre pace and pit loss. That would be unrealistic because Formula 1 strategy is heavily affected by traffic and track position. Model 4 gives the pipeline its on-track racing realism.

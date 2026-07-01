# Model 2 README: Safety Car, Virtual Safety Car and Red Flag Hazard Model

## 1. Purpose

Model 2 predicts the probability of future race interruptions. It estimates the chance of Safety Car, Virtual Safety Car, and Red Flag events over different future horizons.

Model 2 answers this question:

> How likely is a race interruption in the next few laps, and how should that affect strategy?

This matters because pit stops are cheaper under Safety Car or Virtual Safety Car conditions. A strategy that is poor under green-flag racing may become strong if a Safety Car is likely soon.

---

## 2. Main Script

Recommended script name:

```text
train_model2_run_final9.py
```

---

## 3. Sub-Models

Model 2 is made of three event sub-models:

```text
Safety Car hazard model
Virtual Safety Car hazard model
Red Flag hazard model
```

Each event is predicted over multiple horizons:

```text
next 3 laps
next 5 laps
next 10 laps
```

This creates outputs such as:

```text
p_sc_next3
p_sc_next5
p_sc_next10
p_vsc_next3
p_vsc_next5
p_vsc_next10
p_rf_next3
p_rf_next5
p_rf_next10
```

---

## 4. What Each Sub-Part Does

The Safety Car model predicts whether a full Safety Car may happen soon. This is important because a full Safety Car can reduce pit loss and compress the field.

The Virtual Safety Car model predicts whether a VSC may happen soon. A VSC can also reduce pit loss, but usually affects the race differently from a full Safety Car.

The Red Flag model estimates rare race stoppage risk. Since red flags are rare, fallback or prior-based logic may be used when there are not enough positive examples for a stable supervised model.

The horizon structure tells the simulator whether the event risk is immediate, short-term, or medium-term.

---

## 5. Inputs

Important input features include:

```text
race_id
circuit_id
lap_number
race_phase
drivers_running
field spread
gaps between cars
local traffic density
weather indicators
rain flag
previous event flags
track state indicators
```

The model uses current race state features to predict future events. Future event indicators are used only as labels, not as input features.

---

## 6. Machine Learning Method

Model 2 uses probability classification models. Logistic regression and XGBoost are used where enough positive examples exist.

Logistic regression is useful for calibrated probability outputs. XGBoost is useful because event risk can depend on non-linear interactions between race phase, traffic, weather, and circuit characteristics.

For rare events, especially Red Flag, the system may use prior or fallback probabilities to avoid unstable overfitting.

---

## 7. Evaluation Metrics

Model 2 is evaluated using:

```text
ROC-AUC
PR-AUC
Brier score
Log loss
```

ROC-AUC checks event ranking ability. PR-AUC is important because Safety Car and Red Flag events are rare. Brier score checks probability calibration. Log loss penalises confident wrong predictions.

---

## 8. Connection to Other Models

Model 2 outputs are merged into:

```text
outputs/lap_level_with_model1_model2_model3_model4_FINAL_FOR_MODEL5.csv
```

Model 5B uses these probabilities to simulate neutralisation effects and pit stop discounts.

Model 6 indirectly uses Model 2 through Model 5B scenario outcomes.

---

## 9. Why Model 2 Matters

Formula 1 strategy can change completely if a Safety Car or VSC occurs. Model 2 gives the pipeline a probabilistic understanding of race interruption risk. This makes the strategy simulator more realistic than a simple green-flag-only model.
